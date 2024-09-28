import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    
    cv2.imshow('Test', frame)
    print()

    if cv2.waitKey(1) == 27: #27 = esc
        break

cam.release()

cv2.destroyAllWindows()
