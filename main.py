import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget, QApplication, QTextEdit
from PyQt5.QtGui import QImage, QPixmap

#cam = cv2.VideoCapture('rtsp://admin:admin1234@192.168.0.33:554/cam/realmonitor?channel=1&subtype=0')
cam = cv2.VideoCapture(0)
cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
last_center = None
last_check_time = datetime.now()
last_detected_time = datetime.now()
threshold = 1  
missing_threshold = 10  
min_side_length = 5.5
area_limit = 85
distance_threshold = 120
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

def is_star(contour, num_vertices=10, angle_threshold=10, side_length_threshold=0.4):
    epsilon = 0.030 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != num_vertices:
        return False
    
    def angle(pt1, pt2, pt3):
        vec1 = pt1 - pt2
        vec2 = pt3 - pt2
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        return angle

    angles = []
    side_lengths = []
    
    for i in range(num_vertices):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % num_vertices][0]
        p3 = approx[(i + 2) % num_vertices][0]

        side_length = np.linalg.norm(p1 - p2)
        side_lengths.append(side_length)

        angles.append(angle(p1, p2, p3))

    # for angle in angles:
    #     if (angle < 235 - angle_threshold and angle > 235 + angle_threshold) or (angle < 10 - angle_threshold and angle > 10 + angle_threshold):
    #         return False

    max_side = max(side_lengths)
    min_side = min(side_lengths)
    
    if max_side / min_side > 1 + side_length_threshold:
        return False

    return True

def detect_shapes(contours):
    hexagons = []
    stars = []
    for contour in contours:
        epsilon = 0.022 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if is_approximate_hexagon(approx):
            area = cv2.contourArea(contour)
            if area > area_limit:
                hexagons.append(contour)
        elif is_star(contour):
            stars.append(contour)
    return hexagons, stars

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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

def count_unique_positions(centers):
    unique_centers = []
    for center in centers:
        is_unique = True
        for unique_center in unique_centers:
            if np.linalg.norm(np.array(center) - np.array(unique_center)) < threshold:
                is_unique = False
                break
        if is_unique:
            unique_centers.append(center)
    return len(unique_centers)

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
        self.rect_ready = False

        self.region_last_check_times = {}
        self.region_hexagon_positions = {}
        self.region_last_detected_times = {}
        self.region_last_alarm_times = {}
        self.region_last_count_time = {}

        self.is_started = False

        self.log_df = pd.DataFrame(columns=["timestamp", "region", "hexagons", "stars"])

    def closeEvent(self, event):
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_df.to_csv(f"{time_now}.csv", index=False)
        event.accept()


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

        new_region = (max(0, min(self.rx1, self.rx2)), max(0, min(self.ry1, self.ry2)), min(cam_width, max(self.rx1, self.rx2)), min(cam_height, max(self.ry1, self.ry2)))
        
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
                x1, x2 = max(0, min(region[0], region[2])), min(cam_width, max(region[0], region[2]))
                y1, y2 = max(0, min(region[1], region[3])), min(cam_height, max(region[1], region[3]))
                roi = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 5)
                thresh = cv2.adaptiveThreshold(blurred, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detected_hexagons, detected_stars = detect_shapes(contours)

                if region not in self.region_last_check_times:
                    self.region_last_check_times[region] = datetime.now()
                if region not in self.region_last_detected_times:
                    self.region_last_detected_times[region] = datetime.now()
                if region not in self.region_last_alarm_times:
                    self.region_last_alarm_times[region] = datetime.now()
                if region not in self.region_hexagon_positions:
                    self.region_hexagon_positions[region] = []

                hexagons_last_centers = []
                stars_last_centers = []

                for hexagon in detected_hexagons:
                    M = cv2.moments(hexagon)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        hexagons_last_centers.append((cx, cy))
                        cv2.circle(roi, (cx, cy), 3, (0, 0, 255), -1)

                for star in detected_stars:
                    M = cv2.moments(star)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        stars_last_centers.append((cx, cy))
                        cv2.circle(roi, (cx, cy), 2, (255, 255, 0), -1)

                hexagons_unique_centers = count_unique_positions(hexagons_last_centers)
                stars_unique_centers = count_unique_positions(stars_last_centers)

                current_time = time.time()
                if region not in self.region_last_count_time:
                    self.region_last_count_time[region] = current_time

                if current_time - self.region_last_count_time[region] >= 5:
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_log = {"timestamp": time_now, "region": region, "hexagons": hexagons_unique_centers, "stars": stars_unique_centers}
                    self.log_df = self.log_df._append(new_log, ignore_index=True)
                    self.info_box.append(f"Region {region}: Hexagons: {hexagons_unique_centers}, Stars: {stars_unique_centers}")
                    self.region_last_count_time[region] = current_time

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"({x1}, {y1}, {x2}, {y2})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        
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
