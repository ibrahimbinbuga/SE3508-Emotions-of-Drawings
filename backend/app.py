from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
from model.preprocess import extract_image_features
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('./model/final_emotion_recognition_model_vgg16.keras')
with open('./model/model/class_indices.json', 'r') as f:
    class_indices = json.load(f)
inv_map = {v: k for k, v in class_indices.items()}

UPLOAD_FOLDER = './backend/uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0

        raw_uint8 = (img_array[0] * 255).astype(np.uint8)
        feats = extract_image_features(raw_uint8)
        feats[:3] /= 255.0
        feature_array = np.expand_dims(feats, axis=0).astype(np.float32)
        print("Extracted features:", feats)

        preds = model.predict({
            'image_input':   img_array,
            'feature_input': feature_array
        })
        print("Raw predictions:", preds)

        idx = int(np.argmax(preds, axis=1)[0])
        predicted_emotion = inv_map[idx]

        return jsonify({'emotion': predicted_emotion})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)