# src/face_recognition.py
import os
import numpy as np
import cv2
import dlib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def load_employee_data(data_dir):
    images = []
    labels = []
    label_encoder = LabelEncoder()

    # Iterate through each sub-directory in the data directory
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            label = dir_name
            image_files = os.listdir(os.path.join(root, dir_name))
            for image_file in image_files:
                image_path = os.path.join(root, dir_name, image_file)
                # Filter out invalid image files
                if image_path.endswith('.jpg') or image_path.endswith('.png'):
                    images.append(image_path)
                    labels.append(label)

    # Encode the labels
    labels_encoded = label_encoder.fit_transform(labels)

    return images, labels_encoded, label_encoder

def extract_embeddings(images):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    
    embeddings = []
    valid_images = []
    
    for image_path in images:
        # Filter out invalid image files
        if image_path.endswith('.jpg') or image_path.endswith('.png'):
            # Print the current image path for debugging
            print("Processing image:", image_path)
            
            # Read the image
            image = cv2.imread(image_path)
            
            # Check if the image is empty
            if image is None:
                print("Error: Unable to read image at path:", image_path)
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if len(faces) > 0:
                valid_images.append(image_path)
            
            for face in faces:
                landmarks = predictor(gray, face)
                face_embedding = np.array(face_rec_model.compute_face_descriptor(image, landmarks))
                embeddings.append(face_embedding)
    
    print("Number of valid images:", len(valid_images))
    print("Number of detected faces:", len(embeddings))
    
    return embeddings

def train_model(images, labels):
    # Extract facial embeddings
    embeddings = extract_embeddings(images)
    
    # Check if the number of images matches the number of detected faces
    if len(images) != len(embeddings):
        print("Error: Number of images does not match the number of detected faces.")
        return None
    
    # Train the model
    if len(embeddings) == 0:
        print("Error: No faces detected in the provided images.")
        return None
    
    # Determine the maximum number of components
    max_components = min(len(embeddings), len(embeddings[0]))
    n_components = min(100, max_components)
    
    model = make_pipeline(StandardScaler(), PCA(n_components=n_components), SVC(kernel='linear', probability=True))
    model.fit(embeddings, labels)
    
    return model


def save_model(model, model_path):
    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

# Example usage
if __name__ == "__main__":
    data_dir = "data/employee_images"
    model_path = "models/face_detection_model.pkl"
    
    images, labels, label_encoder = load_employee_data(data_dir)
    model = train_model(images, labels)
    save_model(model, model_path)
