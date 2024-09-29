import cv2
import numpy as np
import os
from datetime import datetime, timedelta

log = open("shape_movement_log.txt", mode="a")

pic_num = 0

#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('rtsp://admin:admin1234@192.168.0.33:554/cam/realmonitor?channel=1&subtype=0')

last_center = None
last_check_time = datetime.now()
last_detected_time = datetime.now()
threshold = 15  
missing_threshold = 10  

def is_approximate_hexagon(approx):
    if len(approx) != 6:
        return False
    
    if not cv2.isContourConvex(approx):
        return False
    
    angles = []
    for i in range(6):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % 6][0]
        p3 = approx[(i + 2) % 6][0]
        
        vec1 = p1 - p2
        vec2 = p3 - p2
        
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)
    
    for angle in angles:
        if angle < 55 or angle > 155:
            return False
    
    return True

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 6)

    thresh = cv2.adaptiveThreshold(blurred, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False  

    for contour in contours:
        # Detect
        epsilon = 0.017 * cv2.arcLength(contour, True) # True = Closed Lines
        approx = cv2.approxPolyDP(contour, epsilon, True) # R

        if is_approximate_hexagon(approx):
            area = cv2.contourArea(contour)
            if area > 300:  #by pixel

                # For Area of the contour
                M = cv2.moments(contour) #m00 = pixels in contour -- m10 & m01 for center
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(frame, f"Center: ({cx}, {cy})", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                    current_time = datetime.now()
                    if current_time - last_check_time >= timedelta(seconds=5):
                        if last_center is not None: #if it has been detected even for ones
                            distance = np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2)
                            if distance > threshold:
                                movement_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                                log.write(f"Shape moved at {movement_time} from {last_center} to ({cx}, {cy}), Distance: {distance}\n")
                                print(f"Shape moved at {movement_time} from {last_center} to ({cx}, {cy}), Distance: {distance}")
                                pic_num += 1
                                cv2.imwrite(f"Picture_moved{pic_num}.jpg", frame)

                        last_center = (cx, cy)
                        last_check_time = current_time

                    detected = True
                    last_detected_time = current_time

                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

    if not detected and (datetime.now() - last_detected_time).seconds > missing_threshold:
        cv2.putText(frame, "ALARM: Shape not detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print("ALARM: Shape not detected!")

    frame = cv2.resize(frame, (1360, 750))
    cv2.imshow('Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
log.close()
