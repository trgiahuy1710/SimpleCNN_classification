from flask import Flask, request, jsonify, render_template, send_from_directory
from torchvision import transforms
from PIL import Image
import torch
import io
from torch.nn.functional import softmax
import logging
import os

from SimpleCNN import SimpleCNN

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder='static', template_folder='templates')  # Specify static and template folder paths

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Category labels
categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# Load trained model
try:
    model = SimpleCNN(num_classes=10).to(device)
    checkpoint = torch.load("trained_models/best_cnn.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
except Exception as e:
    logging.error(f"Error while loading the model: {str(e)}")
    raise RuntimeError("Failed to load the model. Ensure the checkpoint file is correct.") from e

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@app.route("/")
def index():
    """Serve the index.html page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Validate that a file is present in the request
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process the uploaded file as an image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({"error": "Invalid image file"}), 400

    # Transform image and move to device
    image = transform(image).unsqueeze(0).to(device)

    try:
        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            probs = softmax(outputs, dim=1)
            max_idx = torch.argmax(probs, dim=1).item()
            predicted_class = categories[max_idx]

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files like JS or CSS if needed."""
    return send_from_directory('static', path)


if __name__ == "__main__":
    # Ensure the directories exist
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # Run the Flask app
    app.run(debug=True)