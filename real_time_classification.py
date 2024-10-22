import cv2
import numpy as np
from flask import Flask, render_template, Response
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import threading

app = Flask(__name__)

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="microorganism_detection_SqueezeNet_50Epochs.tflite")  # Ensure it's a .tflite model
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))

def preprocess_image(frame):
    frame_resized = cv2.resize(frame, (227, 227))  # Resize to match model input
    input_data = frame_resized.astype(np.float32)  # Convert to float32
    input_data /= 255.0  # Normalize to range [0, 1]
    return input_data


def classify_frame(frame):
    input_data = preprocess_image(frame)  # Preprocess frame
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension (1, 227, 227, 3)

    # Set the tensor (input shape should be (1, 227, 227, 3))
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the classification result (assuming it's a classification problem)
    return np.argmax(output_data)

    # Ensure input has the expected shape (227, 227, 3) - remove any extra dimensions
    input_data = np.squeeze(input_data, axis=0)
    
    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the classification result (assuming it's a classification problem)
    return np.argmax(output_data)

def capture_camera():
    picam2.start()
    while True:
        frame = picam2.capture_array()  # Capture image
        result = classify_frame(frame)  # Run classification
        
        # Display result on frame (optional)
        cv2.putText(frame, f"Prediction: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert to JPEG and yield to stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Limit frame rate for classification
        cv2.waitKey(100)

@app.route('/')
def index():
    # Homepage that displays video feed
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=start_flask_app)
    flask_thread.start()

    # Start the camera capture and classification in the main thread
    capture_camera()
