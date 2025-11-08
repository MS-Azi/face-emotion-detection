from flask import Flask, render_template, request, jsonify
from keras.preprocessing import image
import numpy as np
import sqlite3
import os
from datetime import datetime
import tensorflow as tf
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')
print("Database path:", DB_PATH)

try:
    import cv2
    OPENCV_AVAILABLE = True
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
except Exception:
    OPENCV_AVAILABLE = False
    face_cascade = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

TFLITE_MODEL_PATH = "face_emotionModel_compat.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def init_db():
    conn = sqlite3.connect(DB_PATH)
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

def preprocess_image_for_model(img_pil):
    img_gray = img_pil.convert('L').resize((48, 48))
    arr = image.img_to_array(img_gray)
    arr = np.expand_dims(arr, axis=0).astype('float32') / 255.0
    return arr

def detect_emotion_from_pil(img_pil):
    if OPENCV_AVAILABLE:
        try:
            cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = cv_img[y:y+h, x:x+w]
                face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                arr = preprocess_image_for_model(face_pil)
            else:
                arr = preprocess_image_for_model(img_pil)
        except Exception:
            arr = preprocess_image_for_model(img_pil)
    else:
        arr = preprocess_image_for_model(img_pil)

    try:
        interpreter.set_tensor(input_details[0]['index'], arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        if input_details[0]['dtype'] == np.uint8:
            arr_q = (arr * 255).astype(np.uint8)
            interpreter.set_tensor(input_details[0]['index'], arr_q)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
        else:
            raise e

    emotion = emotion_labels[int(np.argmax(output))]
    return emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form.get('name', '')
    email = request.form.get('email', '')
    department = request.form.get('department', '')
    file = request.files.get('image')

    if not file:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    img_pil = Image.open(save_path).convert('RGB')
    emotion = detect_emotion_from_pil(img_pil)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO students (name, email, department, image_path, emotion) VALUES (?, ?, ?, ?, ?)",
                   (name, email, department, save_path, emotion))
    conn.commit()
    conn.close()

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

    return jsonify({
        'success': True,
        'emotion': emotion,
        'message': message,
        'img_path': save_path
    })

@app.route('/webcam_upload', methods=['POST'])
def webcam_upload():
    data = request.get_json()
    if not data or 'imageBase64' not in data:
        return jsonify({'success': False, 'error': 'No image data'}), 400

    b64 = data['imageBase64']
    if ',' in b64:
        b64 = b64.split(',', 1)[1]

    try:
        img_bytes = base64.b64decode(b64)
        img_pil = Image.open(BytesIO(img_bytes)).convert('RGB')
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"webcam_{timestamp}.png"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_pil.save(save_path)

    emotion = detect_emotion_from_pil(img_pil)

    name = data.get('name', '')
    email = data.get('email', '')
    department = data.get('department', '')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO students (name, email, department, image_path, emotion) VALUES (?, ?, ?, ?, ?)",
                   (name, email, department, save_path, emotion))
    conn.commit()
    conn.close()

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

    return jsonify({
        'success': True,
        'emotion': emotion,
        'message': message,
        'img_path': save_path
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
