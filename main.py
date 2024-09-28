import cv2
import numpy as np
from datetime import datetime

log = open("square_detection_log.txt", mode="a")

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray, 50, 150)

    contours, hierarchy_par_child = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) ##############cv2.RETR_LIST

    for contour in contours:
        # Detect
        epsilon = 0.02 * cv2.arcLength(contour, True) # True = Closed Lines
        approx = cv2.approxPolyDP(contour, epsilon, True) # R


        if len(approx) == 6:
            area = cv2.contourArea(contour)
            if area > 1000: #by pixel

                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3) #-1 for all of it , then color , width = 3
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"Square detected at {current_time}\n")
                print(f"Square detected at {current_time}")
                

    cv2.imshow('Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
log.close()
