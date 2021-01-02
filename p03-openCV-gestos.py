import numpy as np
import cv2

cap = cv2.VideoCapture(0)   # '0' argument has to do with camera device selection.
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open cam")
    exit(0)

pt1 = (400, 100)
pt2 = (600, 300)

# Save video output at "./out.avi".
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (frame_width, frame_height))

while (True):
    ret, frame = cap.read()
    if not ret:
        exit(0)

    fgMask = backSub.apply(frame)

    frame = cv2.flip(frame, 1)

##    roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0], : ].copy()
##    cv2.rectangle(frame, pt1, pt2, (255,0,0))
##    cv2.imshow('ROI', roi)
    cv2.imshow('Foreground Mask', fgMask)

    cv2.imshow('frame', frame)
#     out.write(frame)

    keyboard = cv2.waitKey(1)
    if keyboard & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
