# src/face_authentication.py
import os
import cv2
import dlib
import numpy as np
import pickle

def load_model(model_path):
    """
    Load the trained face recognition model.
    Args:
        model_path (str): Path to the trained model file.
    Returns:
        Pipeline: Trained face recognition model.
        LabelEncoder: Label encoder used for encoding employee names.
    """
    print("Loading trained model and label encoder...")
    # Load the trained face recognition model and label encoder
    with open(model_path, 'rb') as f:
        model, label_encoder = pickle.load(f)
    print("Trained model and label encoder loaded successfully.")
    return model, label_encoder

def detect_faces(image):
    """
    Detect faces in the test image.
    Args:
        image (ndarray): Image array.
    Returns:
        list: List of detected faces as rectangles (x, y, w, h).
    """
    print("Detecting faces in the image...")
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Number of detected faces: {len(faces)}")
    print("Faces detected successfully.")
    # Return the coordinates of the detected faces
    return faces

def extract_embeddings(image, detector, predictor, face_rec_model):
    """
    Extract facial embeddings from the detected faces in the test image.
    Args:
        image (ndarray): Image array.
        detector: Dlib face detector object.
        predictor: Dlib shape predictor object.
        face_rec_model: Dlib face recognition model object.
    Returns:
        list: List of facial embeddings.
    """
    print("Extracting facial embeddings...")
    embeddings = []
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        face_embedding = np.array(face_rec_model.compute_face_descriptor(image, landmarks))
        embeddings.append(face_embedding)
    print("Facial embeddings extracted successfully.")
    return embeddings

def authenticate_faces(model, image, label_encoder, detector, predictor, face_rec_model):
    """
    Authenticate faces in the test image using the trained face recognition model.
    Args:
        model (Pipeline): Trained face recognition model.
        image (ndarray): Image array.
        label_encoder (LabelEncoder): Label encoder used for encoding employee names.
        detector: Dlib face detector object.
        predictor: Dlib shape predictor object.
        face_rec_model: Dlib face recognition model object.
    Returns:
        list: List of detected faces as rectangles (x, y, w, h).
        list: List of predicted names for the detected faces.
    """
    print("Authenticating faces in the image...")
    # Detect faces in the test image
    print("Detecting faces in the image...")
    faces = detect_faces(image)
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        return [], []
    
    print(f"Number of detected faces: {len(faces)}")
    
    # Extract facial embeddings for each detected face
    print("Extracting facial embeddings...")
    embeddings = extract_embeddings(image, detector, predictor, face_rec_model)
    
    print(f"Number of extracted embeddings: {len(embeddings)}")
    
    if len(embeddings) == 0:
        print("No facial embeddings extracted.")
        return [], []
    
    # Predict labels for the detected faces
    print("Predicting labels for the detected faces...")
    predicted_labels = model.predict(embeddings)
    
    # Decode the predicted labels
    predicted_names = label_encoder.inverse_transform(predicted_labels)
    
    print("Faces authenticated successfully.")
    return faces, predicted_names

def draw_authentication_result(image, faces, names):
    """
    Draw rectangles around the detected faces and display the authentication result.
    Args:
        image (ndarray): Image array.
        faces (list): List of detected faces as rectangles (x, y, w, h).
        names (list): List of predicted names for the detected faces.
    """
    print("Drawing authentication result...")
    # Draw rectangles around the detected faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Put the predicted name on the image
        cv2.putText(image, names[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the authentication result
    cv2.imshow('Face Authentication Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    model_path = "models/face_recognition_model.pkl"
    test_dir = "data/test"
    
    # Load the trained model and label encoder
    model, label_encoder = load_model(model_path)
    
    print("Initializing face detection models...")
    # Load face detection models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    print("Face detection models initialized successfully.")
    
    # Iterate over all images in the test directory
    print("Iterating over images in the test directory...")
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Read the image
            print(f"Processing image: {filename}")
            image_path = os.path.join(test_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Unable to read image: {filename}")
                continue
            
            print("Image read successfully.")

            # Detect faces in the test image
            faces = detect_faces(image)
            print(f"Faces detected: {faces}")

            # Authenticate faces in the test image
            if len(faces) > 0:  # Check if the list is not empty
                faces, names = authenticate_faces(model, image, label_encoder, detector, predictor, face_rec_model)
                print(f"Names: {names}")

                # Draw the authentication result
                draw_authentication_result(image.copy(), faces, names)
            else:
                print("No faces detected in the image.")
