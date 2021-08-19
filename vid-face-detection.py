import numpy as np
import cv2
import sys


avi = '~/repos/samples/face-detection-exercise/sample.avi'
mp4 = '~/repos/samples/face-detection-exercise/sample.mp4'
# cap = cv2.VideoCapture(mp4)
cap = cv2.VideoCapture(0)

sys.stdout.write(avi)

face_cascade = cv2.CascadeClassifier('haar_face_detection.xml')
eye_cascade = cv2.CascadeClassifier('haar_profile_face_detection.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Print frame height width channels
    height, width, channels = frame.shape
    
    # print height, width, channels
    framestring = frame.tostring()
    sys.stdout.write(framestring)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()