import numpy as np
import cv2
import math

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

cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print ("Unable to open cam")
    exit(0)

ret, bgRef = cap.read()
bgRef = cv2.flip(bgRef, 1)

pt1 = (400, 100)
pt2 = (600, 300)
roiBg = bgRef[pt1[1]:pt2[1],pt1[0]:pt2[0], : ].copy()

roiBg_gray = cv2.cvtColor(roiBg, cv2.COLOR_BGR2GRAY)
roiBg_gray = cv2.GaussianBlur(roiBg_gray, (21, 21), 0)

while (True):
    ret,frame = cap.read()
    if not ret:
	    exit(0)

    frame = cv2.flip(frame, 1)
	
    roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0], : ].copy()

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.GaussianBlur(roi_gray, (21, 21), 0)

    # In each iteration, calculate absolute difference between current frame and reference frame
    difference = cv2.absdiff(roi_gray, roiBg_gray)

    # Apply thresholding to eliminate noise
    thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=0)

    # contornososos
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(roi, contours, -1, (0,255,0),3)

    # HULL COCO
    if (len(contours) > 0):
        hull = cv2.convexHull(contours[0])
        cv2.drawContours(roi, [hull], 0, (255,0,0),3)

        # POGGERS
        cnt = contours[0]
        hull2 = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull2)
          
        if defects is not None:
            for i in range(len(defects)):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                depth = d/256.0
                #print(depth)
                ang = angle(start,end,far)
                cv2.line(roi,start,end,[255,0,0],2)
                cv2.circle(roi,far,5,[0,0,255],-1)

        rect = cv2.boundingRect(cnt)
        p1 = (rect[0], rect[1])
        p2 = (rect[0] + rect[2], rect[1] + rect[3])

        cv2.rectangle(roi, p1, p2, (0, 0, 255), 3)
        pmedio = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        circulo = np.zeros((pt2[0] - pt1[0] , pt2[1] - pt1[1], 1), np.uint8)
        cv2.circle(circulo, pmedio, 65, [255, 255, 255], 1)
        mask = cv2.bitwise_and(thresh, circulo)

        circle_contours, circle_hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
        num_dedos = len(circle_contours) - 1
        
        if num_dedos > 5:
            num_dedos = 5
        if num_dedos < 0:
            num_dedos = 0

        cv2.putText(frame, 'Dedos: ' + str(num_dedos), (460,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,250,0), thickness=4)
        #print(num_dedos)

        cv2.imshow('AND', mask)


    # mostrasion
    cv2.rectangle(frame, pt1, pt2, (255,0,0))
    cv2.imshow('frame', frame)
    cv2.imshow('ROI', roi)
    
    
    keyboard = cv2.waitKey(1)
    if keyboard & 0xFF == ord('q'):
	    break

cap.release()
cv2.destroyAllWindows()
