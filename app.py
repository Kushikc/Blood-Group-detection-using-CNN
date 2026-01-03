import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder to save uploaded files
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = os.path.join(app.root_path, 'best_original_cnn.h5')
print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)

# Blood group classes (change if needed)
BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(image_path, target_size=(224, 224)):
    """Load image, resize, normalize and add batch dimension"""
    img = Image.open(image_path).convert('RGB').resize(target_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # Preprocess and predict
        x = preprocess_image(save_path, target_size=(64,64))  # match model input size
        preds = model.predict(x)
        idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        return jsonify({
            'label': BLOOD_GROUPS[idx],
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
