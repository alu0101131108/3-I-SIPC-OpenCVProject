import numpy as np
import cv2

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open cam")
    exit(0)

pt1 = (400, 100)
pt2 = (600, 300)

while (True):
    ret, frame = cap.read()
    if not ret:
        exit(0)

    frame = cv2.flip(frame, 1)

    roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0], : ].copy()
    cv2.rectangle(frame, pt1, pt2, (255,0,0))
    
    fgMask = backSub.apply(roi,1)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(roi, contours, -1, (0, 255, 0), 3)


    cv2.imshow('ROI', roi)
    cv2.imshow('Foreground Mask', fgMask)
    cv2.imshow('frame', frame)

    keyboard = cv2.waitKey(1)
    if keyboard & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()