import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import time

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v3_small(num_classes=2).to(device)
model.load_state_dict(torch.load('/home/pi/ee347/lab8/best_model.pth', map_location=device))
model.eval()

# Define the face detection classifier and transformations
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Define labels for each class (adjust these to your teammate names or labels)
labels = {0: "Teammate 1", 1: "Teammate 2"}

def capture_and_classify():
    # Open the camera
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Allow camera to warm up

    while True:
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            time.sleep(1)
            continue  # Retry capturing the frame

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # If a face is detected, classify it
            for (x, y, w, h) in faces:
                # Crop and preprocess the face
                face = frame[y:y+h, x:x+w]
                face_tensor = transform(face).unsqueeze(0).to(device)

                # Make a prediction
                with torch.no_grad():
                    output = model(face_tensor)
                    _, predicted = torch.max(output, 1)
                    label = labels[predicted.item()]

                # Draw a rectangle around the face and label it
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Break after detecting and labeling the first face in frame
                break
        else:
            # No face detected, just display the live feed without classification
            cv2.putText(frame, "Waiting for face...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame with or without prediction
        cv2.imshow("Face Recognition", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_classify()
