from flask import Flask, request, jsonify
import pickle
import dlib
import cv2
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
   
    # Load the trained face recognition model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def detect_faces(image):

    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces

def extract_embeddings(image, faces):

    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    
    embeddings = []
    
    for face in faces:
        landmarks = predictor(image, face)
        face_embedding = np.array(face_rec_model.compute_face_descriptor(image, landmarks))
        embeddings.append(face_embedding)
    
    return embeddings

def recognize_person(image, model, label_encoder):
    # Detect faces in the image
    faces = detect_faces(image)
    
    # Extract facial embeddings for each detected face
    embeddings = extract_embeddings(image, faces)
    
    # Predict labels for the detected faces
    predicted_labels = model.predict(embeddings)
    
    # Decode the predicted labels
    predicted_names = label_encoder.inverse_transform(predicted_labels)
    
    # Convert predicted names to strings
    predicted_names = [str(name) for name in predicted_names]
    
    # If multiple faces are detected, return the name of the first recognized person
    if len(predicted_names) > 0:
        return predicted_names[0]
    else:
        return "Unknown"

def load_label_encoder(encoder_path):

    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

@app.route("/recognize", methods=["POST"])
def recognize_face():
    try:
        # Read image from request
        image = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Load the trained model and label encoder
        model = load_model("models/face_recognition.pkl")
        label_encoder = load_label_encoder("models/labels.pkl")
        
        # Recognize faces in the image
        recognized_person_name = recognize_person(image, model, label_encoder)
        
        return jsonify({"recognized_person": recognized_person_name})
    except Exception as e:
        logging.error(f"Error in recognize_face: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=True)
