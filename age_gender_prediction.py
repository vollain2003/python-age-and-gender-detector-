import cv2
import numpy as np
import time

# Define paths to the model files
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"

# Load the models
age_net = cv2.dnn.readNetFromCaffe(age_prototxt, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_prototxt, gender_model)

# Get the names of the output layers for the gender and age networks
gender_output_layer = gender_net.getUnconnectedOutLayersNames()
age_output_layer = age_net.getUnconnectedOutLayersNames()

# Age and gender lists for model output
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Mean values used for image normalization
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Initialize the webcam
video = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Load pre-trained face detector model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("OpenCV version:", cv2.__version__)
print("Gender network layers:", gender_net.getLayerNames())
print("Gender network output layers:", gender_output_layer)
print("Age network layers:", age_net.getLayerNames())
print("Age network output layers:", age_output_layer)

# Print information about the first layer of each network
print("First layer of Gender network:", gender_net.getLayer(0))
print("First layer of Age network:", age_net.getLayer(0))

frame_count = 0
start_time = time.time()

# Modified softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / (e_x.sum(axis=0) + 1e-7)  # Add small epsilon to avoid division by zero

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame. Retrying...")
        time.sleep(1)  # Wait for a second before trying again
        continue

    frame_count += 1
    if frame_count % 30 == 0:  # Print FPS every 30 frames
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected in this frame.")
    else:
        print(f"Detected {len(faces)} face(s) in this frame.")

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = frame[y:y+h, x:x+w]
        
        # Resize face to 227x227 as expected by the model
        face_resized = cv2.resize(face, (227, 227))

        # Prepare input blob for age and gender prediction
        blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward(gender_output_layer)
        gender_preds = softmax(gender_preds[0])  # Use custom softmax function
        print("Raw gender predictions:", gender_preds)

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward(age_output_layer)
        age_preds = softmax(age_preds[0])  # Use custom softmax function
        print("Raw age predictions:", age_preds)

        # Get predicted gender and age
        gender = gender_list[np.argmax(gender_preds)]
        age = age_list[np.argmax(age_preds)]
        print(f"Predicted gender: {gender}, Predicted age: {age}")

        # Draw bounding box and label on the frame
        label = f'{gender}, {age}'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the output
    cv2.imshow("Age and Gender Prediction", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
video.release()
cv2.destroyAllWindows()

print("Program ended.")
