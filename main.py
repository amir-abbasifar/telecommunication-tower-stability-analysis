import cv2
import numpy as np
import os
from datetime import datetime

log = open("square_detection_log.txt", mode="a")

pic_num = int(0)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blurred, 50, 150)

    contours, hierarchy_par_child = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) ##############cv2.RETR_LIST

    for contour in contours:
        # Detect
        epsilon = 0.018 * cv2.arcLength(contour, True) # True = Closed Lines
        approx = cv2.approxPolyDP(contour, epsilon, True) # R


        if len(approx) == 6:
            area = cv2.contourArea(contour)
            if area > 1000: #by pixel
                
                #For Area of the contour
                M = cv2.moments(contour) #m00 = pixels in contour -- m10 & m01 for center
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(frame, f"Center: ({cx}, {cy})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3) #-1 for all of it , then color , width = 3
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.write(f"Square detected at {current_time}\n")
                print(approx)
                print(f"Square detected at {current_time}")

                pic_num = pic_num + 1
                
                # dir = "C:/Users/Amir/Desktop/KarAmoozi/{tarikh}"
                # if not os.path.isdir(mypath):
                #     tarikh = datetime.now().strftime("%Y-%m-%d")
                #     os.makedirs(dir)

                cv2.imwrite(f"Picture{pic_num}.jpg", frame)

    cv2.imshow('Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
log.close()
