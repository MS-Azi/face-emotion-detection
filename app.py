from flask import Flask, render_template, request
from keras.preprocessing import image
import numpy as np
import sqlite3
import os
from datetime import datetime
import tensorflow as tf
from werkzeug.utils import secure_filename


# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="face_emotionModel_compat.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define emotion labels (same as those used in training)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Create or connect to SQLite database
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        email TEXT,
                        department TEXT,
                        image_path TEXT,
                        emotion TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )''')
    conn.commit()
    conn.close()

init_db()

# Route: Home page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    email = request.form['email']
    department = request.form['department']
    file = request.files['image']

    if file:
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        # Preprocess the image for the model
        img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0

        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Get emotion label
        emotion = emotion_labels[np.argmax(predictions)]

        # Store user info + emotion in database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO students (name, email, department, image_path, emotion) VALUES (?, ?, ?, ?, ?)",
                       (name, email, department, img_path, emotion))
        conn.commit()
        conn.close()

        # Friendly message based on emotion
        emotion_messages = {
            'angry': "You look angry. Take a deep breath!",
            'disgust': "You seem displeased. What's bothering you?",
            'fear': "You look scared. Don't worry, you're safe here!",
            'happy': "You’re smiling! Glad to see you happy!",
            'neutral': "You seem calm and composed.",
            'sad': "You look sad. Hope you feel better soon.",
            'surprise': "You look surprised! Something unexpected happened?"
        }

        message = emotion_messages.get(emotion, "Emotion detected.")
        return render_template('index.html', name=name, emotion=emotion, message=message, img_path=img_path)

    else:
        return "No image uploaded."


if __name__ == '__main__':
    app.run()
