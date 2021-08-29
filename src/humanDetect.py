import cv2
import time

video_captured = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade/haarcascade_fullbody.xml')

while (True):
    # read frame-by-frame
    ret, frame = video_captured.read()

    # set the frame to gray as we do not need color, save up the resources
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # pass the frame to the classifier
    persons_detected = classifier.detectMultiScale(gray_frame, 1.3, 5)

    # check if people were detected on the frame
    
    # extract boxes so we can visualize things better
    # for actual deployment with hardware, not needed
    for (x, y, w, h) in persons_detected:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)    
    cv2.imshow('Video footage', frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
