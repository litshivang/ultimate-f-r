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
    unique_labels = []
    label_encoder = LabelEncoder()

    # Iterate through each sub-directory in the data directory
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            label = dir_name
            unique_labels.append(label)  # Collect unique labels
            image_files = os.listdir(os.path.join(root, dir_name))
            for image_file in image_files:
                image_path = os.path.join(root, dir_name, image_file)
                # Filter out invalid image files
                if image_path.endswith('.jpg') or image_path.endswith('.png'):
                    images.append(image_path)
                    labels.append(label)

    # Encode the labels
    labels_encoded = label_encoder.fit_transform(labels)

    return images, labels_encoded, unique_labels, label_encoder


def extract_embeddings(images):
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    
    embeddings = []
    valid_images = []
    labels = []
    
    for image_path in images:
        # Filter out invalid image files
        if image_path.endswith('.jpg') or image_path.endswith('.jpeg') or image_path.endswith('.png'):
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
                
                # Add a label for each detected face in the image
                for face in faces:
                    labels.append(os.path.basename(os.path.dirname(image_path)))  # Use folder name as label
                
                for face in faces:
                    landmarks = predictor(gray, face)
                    face_embedding = np.array(face_rec_model.compute_face_descriptor(image, landmarks))
                    embeddings.append(face_embedding)
    
    print("Number of valid images:", len(valid_images))
    print("Number of detected faces:", len(embeddings))
    print("Number of labels:", len(labels))
    
    return embeddings, valid_images, labels



def train_model(images, labels):
  
    # Extract facial embeddings and labels
    embeddings, _, valid_labels = extract_embeddings(images)
    
    # Check if the number of embeddings matches the number of labels
    if len(embeddings) != len(valid_labels):
        print("Error: Number of embeddings does not match the number of labels.")
        return None, None
    
    # Train the model
    if len(embeddings) == 0:
        print("Error: No faces detected in the provided images.")
        return None, None
    
    # Determine the maximum number of components
    max_components = min(len(embeddings), len(embeddings[0]))
    n_components = min(100, max_components)
    
    # Create and train the model
    model = make_pipeline(StandardScaler(), PCA(n_components=n_components), SVC(kernel='linear', probability=True))
    model.fit(embeddings, valid_labels)
    
    # Create and fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_labels)
    
    return model, label_encoder


def save_model(model, label_encoder, model_path, encoder_path):

    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the label encoder
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    data_dir = "data/employee_images"
    model_path = "models/face_recognition_model.pkl"
    encoder_path = "models/label_encoder.pkl"
    
    # Load employee data
    images, labels, unique_labels, label_encoder = load_employee_data(data_dir)
    
    # Train the face recognition model using only the valid images
    model, _ = train_model(images, labels)
    
    if model is not None:
        # Save the trained model and label encoder
        save_model(model, label_encoder, model_path, encoder_path)  # Pass only model and label_encoder
        print("Model and label encoder saved successfully.")
    else:
        print("Failed to train the model.")