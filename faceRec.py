import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
train_dir = r"C:\Users\MD HASSAN IRSHAD\Desktop\Face regonization system\dataset\train"
test_dir = r"C:\Users\MD HASSAN IRSHAD\Desktop\Face regonization system\dataset\test"

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Initialize LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# # Function: Prepare training data
def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_dict = {}
    current_label = 0
    
    for person_name in os.listdir(data_folder_path):
        person_path = os.path.join(data_folder_path, person_name)
        if not os.path.isdir(person_path):
            continue
        
        print(f"Processing person: {person_name}")
        label_dict[current_label] = person_name
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            # Read image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Detect faces
            faces_rects = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces_rects:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                faces.append(face)
                labels.append(current_label)
                break  # Use only one face per image
            
        current_label += 1
    
    return faces, labels, label_dict




# # Prepare training data
print("Preparing training data...")
faces, labels, label_dict = prepare_training_data(train_dir)
print(f"Total faces: {len(faces)}, Total labels: {len(set(labels))}")

# # Train recognizer
print("Training recognizer...")
recognizer.train(faces, np.array(labels))
print("Training complete!")


# Function: Evaluate on test data
def evaluate(test_folder):
    correct = 0
    total = 0
    for person_name in os.listdir(test_folder):
        person_path = os.path.join(test_folder, person_name)
        if not os.path.isdir(person_path):
            continue
        
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            faces_rects = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rects:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                label_pred, confidence = recognizer.predict(face)
                predicted_name = label_dict[label_pred]
                
                print(f"Actual: {person_name} | Predicted: {predicted_name} | Confidence: {confidence:.2f}")
                if predicted_name == person_name:
                    correct += 1
                total += 1
                break  # Only one face per image
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nRecognition Accuracy: {accuracy:.2f}")



def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found.")
        return

    faces_rects = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces_rects) == 0:
        print("No face detected.")
        return

    for (x, y, w, h) in faces_rects:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        label_pred, confidence = recognizer.predict(face)
        predicted_name = label_dict[label_pred]
        print(f"Predicted: {predicted_name} | Confidence: {confidence:.2f}")

        # Show the image
        img_color = cv2.imread(image_path)
        cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_color, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (36,255,12), 2)
        cv2.imshow("Prediction", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break





print("\n[INFO] Evaluating on test data...")
evaluate(test_dir)



# Optional: Predict on your own image
user_input = input("\nEnter path to your own image for prediction (or press Enter to skip): ").strip()
if user_input:
    predict_image(user_input)






img = cv2.imread(user_input)
img.shape
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image.shape

face = face_cascade.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')

