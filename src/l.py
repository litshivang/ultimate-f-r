import pickle

def load_label_encoder(encoder_path):
    """
    Load the label encoder used during training.
    Args:
        encoder_path (str): Path to the label encoder file.
    Returns:
        object: Label encoder object.
    """
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder


# Load the label encoder
label_encoder = load_label_encoder("models/label_encoder.pkl")

# Get the unique labels
unique_labels = label_encoder.classes_

# Print the mapping between labels and names
for label in unique_labels:
    print(f"Label {label}: {label_encoder.inverse_transform([label])[0]}")
