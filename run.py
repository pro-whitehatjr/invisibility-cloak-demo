from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    #Allowing the webcam to start by making the code sleep for 2 seconds

    bg = 0
    time.sleep(5)

    cap = cv2.VideoCapture(0)
    
    #Capturing background for 60 frames
    for i in range(60):
        ret, bg = cap.read()
    #Flipping the background
    bg = np.flip(bg, axis=1)

    #Reading the captured frame until the camera is open
    while (cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break
        #Flipping the image for consistency
        img = np.flip(img, axis=1)

        #Converting the color from BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #Generating mask to detect red colour
        #These values can also be changed as per the color
        lower_red = np.array([0, 120, 50])
        upper_red = np.array([10, 255,255])
        mask_1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask_2 = cv2.inRange(hsv, lower_red, upper_red)

        mask_1 = mask_1 + mask_2

        #Open and expand the image where there is mask 1 (color)
        mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        #Selecting only the part that does not have mask one and saving in mask 2
        mask_2 = cv2.bitwise_not(mask_1)

        #Keeping only the part of the images without the red color
        #(or any other color you may choose)
        res_1 = cv2.bitwise_and(img, img, mask=mask_2)

        #Keeping only the part of the images with the red color
        #(or any other color you may choose)
        res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

        #Generating the final output by merging res_1 and res_2
        final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
        ret, buffer = cv2.imencode('.jpg', final_output)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()
