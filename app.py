from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
import datetime
import sqlite3
import socket
from cryptography.fernet import Fernet

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Encryption key generation (this should be securely stored and reused in real applications)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# SQLite setup
DATABASE = 'videos.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            encryption_key BLOB NOT NULL,
            upload_time TIMESTAMP NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Color ranges for HSV color detection
color_ranges = {
    "Red": ((0, 100, 100), (10, 255, 255)),
    "Green": ((40, 40, 40), (80, 255, 255)),
    "Blue": ((100, 100, 100), (130, 255, 255))
}

# Overlay timestamp on video frame
def overlay_timestamp(frame, timestamp):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner_of_text = (10, frame.shape[0] - 10)
    font_scale = 1
    font_color = (0, 255, 0)
    line_type = 2

    cv2.putText(frame, f'Time: {timestamp:.2f} sec',
                bottom_left_corner_of_text,
                font,
                font_scale,
                font_color,
                line_type)

# Crop video based on color detection
def crop_video_on_color(input_video_path, output_video_path, target_color):
    if target_color not in color_ranges:
        print("Error: Invalid target color. Options are 'Red', 'Green', or 'Blue'.")
        return

    target_lower, target_upper = color_ranges[target_color]

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, target_lower, target_upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cv2.countNonZero(mask) > 0:
            overlay_timestamp(frame, current_time)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Detect face in video based on input image
def detect_and_save(video_path, image_path, output_video_name):
    cap = cv2.VideoCapture(video_path)
    image_to_detect = cv2.imread(image_path)
    if image_to_detect is None:
        print("Error: Could not read the image file.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image_to_detect = cv2.cvtColor(image_to_detect, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image_to_detect, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("Error: No face detected in the provided image.")
        return

    (x, y, w, h) = faces[0]
    target_face_region = gray_image_to_detect[y:y+h, x:x+w]

    out = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        target_face_detected = False

        for (x_face, y_face, w_face, h_face) in faces:
            current_face_region = gray_frame[y_face:y_face+h_face, x_face:x_face+w_face]
            resized_current_face = cv2.resize(current_face_region, (w, h))
            diff = cv2.absdiff(target_face_region, resized_current_face)
            mean_diff = diff.mean()

            if mean_diff < 50:
                target_face_detected = True
                cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
                overlay_timestamp(frame, current_time)

        if target_face_detected:
            if out is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))
            out.write(frame)
            frame_count += 1

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        print("Recognition completed. Output video saved as", output_video_name)
    else:
        print("No frames with the target person detected. No output video created.")

# Encrypt and store video details in the database
def save_to_database(filename, key):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    encrypted_key = cipher_suite.encrypt(key)
    cursor.execute('INSERT INTO processed_videos (filename, encryption_key, upload_time) VALUES (?, ?, ?)', 
                   (filename, encrypted_key, datetime.datetime.now()))
    conn.commit()
    conn.close()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    
    video = request.files['video']
    detection_type = request.form['detectionType']
    
    if video.filename == '':
        return redirect(request.url)
    
    filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '_' + video.filename
    input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(input_video_path)

    output_filename = 'processed_' + filename
    output_video_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

    if detection_type == "Color":
        target_color = request.form['color']
        crop_video_on_color(input_video_path, output_video_path, target_color)
    else:
        face_image = request.files['faceImage']
        face_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_' + face_image.filename)
        face_image.save(face_image_path)
        detect_and_save(input_video_path, face_image_path, output_video_path)

    # Save video information to the database
    save_to_database(output_filename, key)

    return redirect(url_for('success'))

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Find an available port for the app to run on
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 0))
    addr, port = s.getsockname()
    s.close()
    return port

if __name__ == "__main__":
    port = int(os.environ.get("PORT", find_free_port()))
    app.run(host="0.0.0.0", port=port, debug=True)
