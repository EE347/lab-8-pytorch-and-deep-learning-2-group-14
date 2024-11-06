from picamera2 import Picamera2
import cv2
import numpy as np
import os
from datetime import datetime

# Function to create a unique filename based on the current timestamp
def generate_filename(prefix, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

# Create directories if they don't exist
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the camera
picam2.start()

# Initialize counters for the number of images
image_count = 0
train_count = 0
test_count = 0

while image_count < 60:
    # Capture a frame
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 0)  # Flip the frame vertically

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Process the detected faces
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y + h, x:x + w]

        # Resize the face to 64x64
        face_resized = cv2.resize(face, (64, 64))

        # Save 10 images to the 'test' folder, and 50 to the 'train' folder
        if image_count < 10:
            image_filename = os.path.join('test', generate_filename('test_image' + str(image_count), 'jpg'))
            test_count += 1
        else:
            image_filename = os.path.join('train', generate_filename('train_image' + str(image_count), 'jpg'))
            train_count += 1

        # Save the cropped and resized face
        cv2.imwrite(image_filename, face_resized)
        print(f"Captured and saved: {image_filename}")

        # Increment the image count
        image_count += 1

        # Display the cropped face (optional)
        cv2.imshow("Cropped Face", face_resized)

    # Display the frame with face rectangles (optional)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Camera Feed", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
picam2.close()
cv2.destroyAllWindows()

print(f"Total images saved to 'test' folder: {test_count}")
print(f"Total images saved to 'train' folder: {train_count}")
