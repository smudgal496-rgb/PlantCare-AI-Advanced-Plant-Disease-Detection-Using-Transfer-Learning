import os
import secrets
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, session, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# --- 1. DIRECTORY SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['STATIC_RESULT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'images')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_RESULT_FOLDER'], exist_ok=True)

# --- 2. MODEL LOADING ---
# Replace these with your actual labels in alphabetical order!
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy'] 

MODEL_PATH = os.path.join(BASE_DIR, 'model.keras')
model = None

try:
    if os.path.exists(MODEL_PATH):
        # We load with compile=False to fix Dense layer input issues
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ SUCCESS: Model loaded successfully.")
        # Debugging: Print input shape to terminal
        print("Model expects input shape:", model.input_shape)
    else:
        print(f"❌ ERROR: 'model.keras' not found in {BASE_DIR}")
except Exception as e:
    print(f"❌ ERROR LOADING MODEL: {e}")

# --- 3. PREDICTION LOGIC ---
def predict_image(file_path):
    # 1. Load and resize image to 224x224 (Standard for most Keras models)
    img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    
    # 2. Add Batch Dimension -> Shape becomes (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 3. Normalization (Convert 0-255 to 0-1 range)
    img_array = img_array.astype('float32') / 255.0

    # 4. Run Prediction with Error Handling for "Two Inputs"
    try:
        # Standard approach
        predictions = model.predict(img_array)
    except Exception:
        # Fallback: wrap in a list if the model expects multiple inputs
        predictions = model.predict([img_array])

    # 5. Process results
    # Use softmax to get probabilities
    score = tf.nn.softmax(predictions[0])
    result_index = np.argmax(score)
    
    label = CLASS_NAMES[result_index]
    confidence = 100 * np.max(score)
    
    return label, f"{confidence:.2f}%"

# --- 4. ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    if file and model:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        try:
            label, confidence = predict_image(temp_path)

            unique_name = f"res_{secrets.token_hex(4)}.jpg"
            static_path = os.path.join(app.config['STATIC_RESULT_FOLDER'], unique_name)
            
            with Image.open(temp_path) as img:
                img.convert('RGB').save(static_path)

            session['prediction'] = label
            session['confidence'] = confidence
            session['image_url'] = unique_name

            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        return jsonify({'success': False, 'error': 'Model not initialized'}), 500

@app.route('/result')
def result():
    return render_template('result.html', 
                           prediction=session.get('prediction'),
                           confidence=session.get('confidence'),
                           image_url=session.get('image_url'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)