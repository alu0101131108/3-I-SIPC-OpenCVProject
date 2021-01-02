import numpy as np
import cv2
import math

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open cam")
    exit(0)

pt1 = (400, 100)
pt2 = (600, 300)

def angle(s,e,f):
    v1 = [s[0]-f[0],s[1]-f[1]]
    v2 = [e[0]-f[0],e[1]-f[1]]
    ang1 = math.atan2(v1[1],v1[0])
    ang2 = math.atan2(v2[1],v2[0])
    ang = ang1 - ang2
    if (ang > np.pi):
        ang -= 2*np.pi
    if (ang < -np.pi):
        ang += 2*np.pi
    return ang*180/np.pi

while (True):
    ret, frame = cap.read()
    if not ret:
        exit(0)

    frame = cv2.flip(frame, 1)

    roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0], : ].copy()
    cv2.rectangle(frame, pt1, pt2, (255,0,0))
    
    backSub.apply(roi,1)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(roi, contours, -1, (0, 255, 0), 3)

    if len(contours) != 0:
        cnt = contours[0]
        hull = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull)

        # error en defects por algun motivo estraño a veces es lista a veces no

        if isinstance(defects, list):

            for i in range(len(defects)):
                s, e, f, d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                depth = d/256.0
                print(depth)
                ang = angle(start,end,far)
                cv2.line(roi, start, end, [255,0,0],2)
                cv2.circle(roi, far, 5, [0,0,255], -1)

    cv2.imshow('ROI', roi)
    cv2.imshow('frame', frame)
    
    keyboard = cv2.waitKey(1)
    if keyboard & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()