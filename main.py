import cv2
import numpy as np
from datetime import datetime, timedelta
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget, QApplication, QTextEdit
from PyQt5.QtGui import QImage, QPixmap

log = open("shape_movement_log.txt", mode="a")
pic_num = 0
#cam = cv2.VideoCapture('rtsp://admin:admin1234@192.168.0.33:554/cam/realmonitor?channel=1&subtype=0')
cam = cv2.VideoCapture(0)
last_center = None
last_check_time = datetime.now()
last_detected_time = datetime.now()
threshold = 50  
missing_threshold = 10  
min_side_length = 5.5
area_limit = 85
distance_threshold = 50
movement_check_interval = 5
hexagon_positions = []
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
            if area > area_limit:
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

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shape Detector")
        self.setGeometry(100, 100, 1000, 800)

        self.video_label = QLabel(self)
        self.video_label.setGeometry(20, 20, 640, 480)

        self.start_button = QPushButton("Start", self)
        self.start_button.setGeometry(700, 100, 100, 40)
        self.start_button.clicked.connect(self.start_video)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setGeometry(700, 160, 100, 40)
        self.stop_button.clicked.connect(self.stop_video)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setGeometry(700, 220, 100, 40)
        self.exit_button.clicked.connect(self.close)

        self.regions = []

        self.last_alarm_time = datetime.now()


        self.add_button = QPushButton("Add Region", self)
        self.add_button.setGeometry(700, 280, 100, 40)
        self.add_button.setEnabled(False) 
        self.add_button.clicked.connect(self.add_region)

        self.delete_button = QPushButton("Delete Region", self)
        self.delete_button.setGeometry(700, 340, 100, 40)
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_region)

        self.info_box = QTextEdit(self)
        self.info_box.setGeometry(20, 520, 760, 250)
        self.info_box.setReadOnly(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.drawing = False
        self.ix, self.iy = -1, -1
        self.rx1, self.ry1, self.rx2, self.ry2 = 0, 0, 0, 0

        self.last_check_time = datetime.now()
        self.last_detected_time = datetime.now()
        self.hexagon_positions = []
        self.rect_ready = False

        self.region_last_check_times = {} 
        self.region_hexagon_positions = {} 
        self.region_last_detected_times = {} 
        self.region_last_alarm_times = {} 

        self.is_started = False


    def start_video(self):
        self.is_started = True
        self.timer.start(30)
        self.add_button.setEnabled(True) 
        self.delete_button.setEnabled(True)

    def stop_video(self):
        self.is_started =False
        self.timer.stop()
        self.add_button.setEnabled(False)
        self.delete_button.setEnabled(False)

    def mousePressEvent(self, event):
        if self.is_started:
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.ix, self.iy = event.x(), event.y()
                self.rx1, self.ry1 = self.ix, self.iy
                self.rx2, self.ry2 = self.ix, self.iy
                self.delete_button.setEnabled(False)
                self.update_frame()

    def mouseMoveEvent(self, event):
        if self.is_started:
            if self.drawing:
                self.rx2, self.ry2 = event.x(), event.y()
                self.rect_ready = True
                self.update_frame()

    def mouseReleaseEvent(self, event):
        if self.is_started:
            if event.button() == Qt.LeftButton:
                self.drawing = False
                self.rx2, self.ry2 = event.x(), event.y()
                self.rect_ready = True
                self.delete_button.setEnabled(True)
                self.update_frame()

    def add_region(self):
        if self.rx1 == self.rx2 or self.ry1 == self.ry2:
            self.info_box.append("No region selected!")
            return

        new_region = (min(self.rx1, self.rx2), min(self.ry1, self.ry2), max(self.rx1, self.rx2), max(self.ry1, self.ry2))
        
        for region in self.regions:
            if (abs(new_region[0] - region[0]) < 5 and 
                abs(new_region[1] - region[1]) < 5 and 
                abs(new_region[2] - region[2]) < 5 and 
                abs(new_region[3] - region[3]) < 5):
                self.info_box.append("Region already added!")
                return

        self.regions.append(new_region)
        self.info_box.append(f"Region added: {new_region}")

        self.rx1 = self.rx2 = self.ry1 = self.ry2 = 0
        self.update_frame()


    def delete_region(self):
        if self.regions:
            self.regions.pop()
            self.info_box.append("Last region deleted.")
            self.add_button.setEnabled(True)
            self.rect_ready = False
            self.update_frame()
        
        self.rx1, self.ry1, self.rx2, self.ry2 = 0, 0, 0, 0
        self.rect_ready = False

    def update_frame(self):
        ret, frame = cam.read()
        if ret:
            for region in self.regions:
                x1, x2 = min(region[0], region[2]), max(region[0], region[2])
                y1, y2 = min(region[1], region[3]), max(region[1], region[3])

                if x1 < x2 and y1 < y2:
                    roi = frame[y1:y2, x1:x2]
                else:
                    roi = frame

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 5)
                thresh = cv2.adaptiveThreshold(blurred, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detected_hexagons, detected_circles = detect_hexagons_and_circles(contours)
                nearby_circles = find_nearby_circles(detected_hexagons, detected_circles, distance_threshold)

                current_positions = []
                for hexagon in detected_hexagons:
                    M = cv2.moments(hexagon)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        current_positions.append((cx, cy))
                        cv2.circle(roi, (cx, cy), 4, (255, 0, 0), -1)

                if region not in self.region_last_check_times:
                    self.region_last_check_times[region] = datetime.now()

                if region not in self.region_hexagon_positions:
                    self.region_hexagon_positions[region] = current_positions

                if region not in self.region_last_detected_times:
                    self.region_last_detected_times[region] = datetime.now()

                if region not in self.region_last_alarm_times:
                    self.region_last_alarm_times[region] = datetime.now()

                if datetime.now() - self.region_last_check_times[region] > timedelta(seconds=movement_check_interval):
                    if check_movement(current_positions, self.region_hexagon_positions[region], threshold):
                        self.info_box.append(f"Shape Moved in region {region}!")

                    self.region_hexagon_positions[region] = current_positions
                    self.region_last_check_times[region] = datetime.now()

                for circle in nearby_circles:
                    M = cv2.moments(circle)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        circle_center = (cx, cy)
                        cv2.circle(roi, (cx, cy), 2, (0, 255, 0), -1)

                if not detected_hexagons:
                    if (datetime.now() - self.region_last_detected_times[region]).seconds > missing_threshold:
                        if (datetime.now() - self.region_last_alarm_times[region]).seconds > missing_threshold:
                            self.info_box.append(f"ALARM: Shape not detected in region {region}!")
                            self.region_last_alarm_times[region] = datetime.now()
                else:
                    self.region_last_detected_times[region] = datetime.now()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f"({x1}, {y1}, {x2}, {y2})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text, (x1, y1 - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
        if self.drawing or (self.rx1 != 0 and self.ry1 != 0 and self.rx2 != 0 and self.ry2 != 0):
            cv2.rectangle(frame, (self.rx1, self.ry1), (self.rx2, self.ry2), (0, 255, 0), 2)

            text = f"({self.rx1}, {self.ry1}, {self.rx2}, {self.ry2})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (self.rx1, self.ry1 - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


        qformat = QImage.Format_RGB888
        out_image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(out_image))

    def is_within_regions(self, x, y):
        for region in self.regions:
            if region[0] <= x <= region[2] and region[1] <= y <= region[3]:
                return True
        return False

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
