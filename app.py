from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required to use Flask session

# Load the pre-trained model
model = load_model('my_cnn_model_retrained1_vgg.h5')

# Define folder to store uploaded images
UPLOAD_FOLDER = "C:/Users/benja/Documents/mini project/predicted images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Preprocessing function for images
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((128, 128))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def patient_details():
    return render_template('patient_details.html')

@app.route('/upload', methods=['POST'])
def upload_page():
    # Save patient details to session
    session['name'] = request.form['name']
    session['age'] = request.form['age']
    session['gender'] = request.form['gender']
    
    return render_template('upload_ecg.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('upload_page'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_page'))

    if file:
        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image and predict using the model
        processed_image = preprocess_image(filepath)
        prediction = model.predict(processed_image)

        # Convert prediction to human-readable form
        result = 'Myocardial Infarction Detected' if prediction[0][0] > 0.5 else 'Normal ECG'

        return render_template('result.html', prediction=result, name=session.get('name'), age=session.get('age'), gender=session.get('gender'))

if __name__ == '__main__':
    app.run(debug=True)

