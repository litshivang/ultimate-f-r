# Load existing model and label encoder
def load_existing_model(model_path, encoder_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

# Update model with new employee
def update_model_with_new_employee(model, label_encoder, new_images, new_labels):
    # Extract embeddings for new images
    new_embeddings = extract_embeddings(new_images)
    
    # Append new embeddings and labels to existing data
    existing_embeddings, existing_labels = load_existing_embeddings_and_labels()
    updated_embeddings = existing_embeddings + new_embeddings
    updated_labels = existing_labels + new_labels
    
    # Retrain the model using updated data
    model.fit(updated_embeddings, updated_labels)
    
    return model

# Save updated model and label encoder
def save_updated_model(model, label_encoder, model_path, encoder_path):
    save_model(model, label_encoder, model_path, encoder_path)
    print("Updated model saved successfully.")

# Example usage
if __name__ == "__main__":
    data_dir_new_employee = "data/new_employee_images"
    model_path = "models/face_recognition_model.pkl"
    encoder_path = "models/label_encoder.pkl"
    
    # Load existing model and label encoder
    model, label_encoder = load_existing_model(model_path, encoder_path)
    
    # Load images and labels for new employee
    new_images, new_labels, _, _ = load_employee_data(data_dir_new_employee)
    
    # Update model with new employee
    updated_model = update_model_with_new_employee(model, label_encoder, new_images, new_labels)
    
    # Save updated model and label encoder
    save_updated_model(updated_model, label_encoder, model_path, encoder_path)
