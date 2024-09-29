import cv2
import numpy as np
import os
from datetime import datetime, timedelta

log = open("shape_movement_log.txt", mode="a")

pic_num = 0

cam = cv2.VideoCapture('rtsp://admin:admin1234@192.168.0.33:554/cam/realmonitor?channel=1&subtype=0')
#cam = cv2.VideoCapture(0)

last_center = None
last_check_time = datetime.now()
last_detected_time = datetime.now()
threshold = 50  
missing_threshold = 10  
min_side_length = 5.5
area_limit = 10
distance_threshold = 50  # Maximum distance for detecting a circle near hexagons
movement_check_interval = 5  # Seconds between movement checks
hexagon_positions = []  # To store positions of hexagons
drawing = False
ix, iy = -1, -1
rx1, ry1, rx2, ry2 = 0, 0, 0, 0

def is_approximate_hexagon(approx):
    if len(approx) != 6:
        return False
    
    if not cv2.isContourConvex(approx):
        return False
    
    angles = []
    side_lengths = []

    for i in range(6):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % 6][0]
        p3 = approx[(i + 2) % 6][0]

        side_length = np.linalg.norm(p1 - p2)
        side_lengths.append(side_length)

        vec1 = p1 - p2
        vec2 = p3 - p2

        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)

    for angle in angles:
        if angle < 55 or angle > 155:
            return False
        
    for length in side_lengths:
        if length < min_side_length:
            return False
        
    return True

def is_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * (radius ** 2)
    return abs(circle_area - area) / circle_area < 0.22 # Adjust tolerance as needed 

def detect_hexagons_and_circles(contours):
    hexagons = []
    circles = []
    for contour in contours:
        epsilon = 0.022 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True) # True = Closed Lines

        if is_approximate_hexagon(approx):
            area = cv2.contourArea(contour)
            if area > area_limit: # by pixel
                hexagons.append(contour)

        elif is_circle(contour):
            circles.append(contour)

    return hexagons, circles

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_nearby_circles(hexagons, circles, distance_threshold):
    nearby_circles = []
    for circle in circles:
        M = cv2.moments(circle)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            circle_center = (cx, cy)

            for hexagon in hexagons:
                M_hex = cv2.moments(hexagon)
                if M_hex["m00"] != 0:
                    hx = int(M_hex["m10"] / M_hex["m00"])
                    hy = int(M_hex["m01"] / M_hex["m00"])
                    hexagon_center = (hx, hy)

                    distance = calculate_distance(circle_center, hexagon_center)
                    if distance < distance_threshold:
                        nearby_circles.append(circle)
                        break
    return nearby_circles

def check_movement(new_positions, old_positions, threshold):
    for new_pos in new_positions:
        moved = True
        for old_pos in old_positions:
            if calculate_distance(new_pos, old_pos) < threshold:
                moved = False
                break
        if moved:
            return True
    return False

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rx1, ry1, rx2, ry2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rx1, ry1, rx2, ry2 = ix, iy, x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx1, ry1, rx2, ry2 = ix, iy, x, y

cv2.namedWindow("Test")
cv2.setMouseCallback("Test", draw_rectangle)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Define the region
    height, width = frame.shape[:2]
    
    # Region Should be valid
    if drawing or (rx1 != 0 and ry1 != 0 and rx2 != 0 and ry2 != 0):
        rx1, rx2 = max(0, rx1), min(width, rx2)
        ry1, ry2 = max(0, ry1), min(height, ry2)
        if rx2 > rx1 and ry2 > ry1:
            region = frame[ry1:ry2, rx1:rx2]
        else:
            region = frame
    else:
        region = frame

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 5)
    #median = cv2.medianBlur(blurred, 3)
    #bilateral = cv2.bilateralFilter(median, 3, 50, 50)
    thresh = cv2.adaptiveThreshold(blurred, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_hexagons, detected_circles = detect_hexagons_and_circles(contours)
    nearby_circles = find_nearby_circles(detected_hexagons, detected_circles, distance_threshold)

    current_positions = []
    for hexagon in detected_hexagons:
        M = cv2.moments(hexagon)  #m00 = pixels in contour -- m10 & m01 for center
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            current_positions.append((cx + rx1, cy + ry1))  
            cv2.circle(frame, (cx + rx1, cy + ry1), 4, (255, 0, 0), -1)
            #cv2.drawContours(frame, [hexagon], -1, (0, 255, 0), 3)

            #last_detected_time = datetime.now()

    if datetime.now() - last_check_time > timedelta(seconds=movement_check_interval):
        if hexagon_positions and check_movement(current_positions, hexagon_positions, threshold):
            print("Shape Moved!")

        hexagon_positions = current_positions
        last_check_time = datetime.now()

    for circle in nearby_circles:
        M = cv2.moments(circle)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            circle_center = (cx + rx1, cy + ry1)
            cv2.circle(frame, (cx + rx1, cy + ry1), 2, (0, 255, 0), -1)
            #cv2.drawContours(frame, [circle], -1, (0, 0, 255), 3)

    if not detected_hexagons and (datetime.now() - last_detected_time).seconds > missing_threshold:
        cv2.putText(frame, "ALARM: Shape not detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if drawing or (rx1 != 0 and ry1 != 0 and rx2 != 0 and ry2 != 0):
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

    #frame = cv2.resize(frame, (1300, 700))
    cv2.imshow('Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
log.close()
