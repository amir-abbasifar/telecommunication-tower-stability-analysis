import cv2
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > 1000: 
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

    cv2.imshow('Test', frame)

    if cv2.waitKey(1):
        break

cam.release()
cv2.destroyAllWindows()
