from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json  
from tensorflow.keras.preprocessing import image  
from tensorflow.keras import models
from record import gen_record

trained_model = models.load_model('../trained_models/trained_vggface.h5', compile=False)

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  

app = Flask(__name__)

camera = cv2.VideoCapture(0)
# count = 0
def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        success, frame = camera.read()
        if not success:
            break
        else:
            color_img= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        
            faces_detected = face_haar_cascade.detectMultiScale(color_img, 1.32, 5)  
            
            for (x,y,w,h) in faces_detected:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
                roi_gray=color_img[y:y+w,x:x+h]          #cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(96,96))  
                img_pixels = image.img_to_array(roi_gray)  
                img_pixels = np.expand_dims(img_pixels, axis = 0)  
                predictions = trained_model.predict(img_pixels)  
        
                #find max indexed array  
                
                max_index = np.argmax(predictions[0])  
        
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  
                predicted_emotion = emotions[max_index]  
                # if predicted_emotion == 'angry':
                    # count+=1
                    # print("FACE ANGRY")
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)  
        
            resized_img = cv2.resize(frame, (1000, 700))  
            
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_feed')
def audio_feed():
    # record = gen_record()
    # if record:
    #     print("VOICE ANGRY")
        # count += 1
    return Response(gen_record(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def score():
#     return Response(count, mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)