from flask import Flask, redirect, url_for, render_template, request, Response
import cv2
import numpy as np
import tensorflow as tf
app=Flask(__name__)
camera=cv2.VideoCapture(0)

model=tf.keras.models.load_model('model.keras')
cv2.ocl.setUseOpenCL(False)
emotion_dict={0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}

def detect_emotion(frame):
    # Find haar cascade to draw bounding box around face
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces=face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        cropped_img=np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction=model.predict(cropped_img)
        max_index=int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[max_index], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame

def gen_frames():
    while True:
        success, frame=camera.read()
        if not success:
            break
        else:
            frame=detect_emotion(frame)
            ret, buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method=='POST':
        camera.release()
        return render_template('exitpage.html')
    

if __name__=='__main__':
    app.run()