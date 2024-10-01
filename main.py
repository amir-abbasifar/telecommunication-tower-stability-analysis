import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget, QApplication, QTextEdit, QSlider, QSpinBox, QLineEdit
from PyQt5.QtGui import QImage, QPixmap

#cam = cv2.VideoCapture('rtsp://admin:admin1234@192.168.0.33:554/cam/realmonitor?channel=1&subtype=0')
cam = cv2.VideoCapture(0)
cam_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

min_side_length = 2
threshold_of_fine_unique_pos = 10
area_limit = 50

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
        if angle < 40 or angle > 170:
            return False
    for length in side_lengths:
        if length < min_side_length:
            return False
    return True

def detect_shapes(contours, epsilon_val):
    hexagons = []
    for contour in contours:
        epsilon = epsilon_val * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if is_approximate_hexagon(approx):
            area = cv2.contourArea(contour)
            if area > area_limit:
                hexagons.append(contour)
    return hexagons

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def check_movement(new_positions, old_positions, given_thresh):
    for new_pos in new_positions:
        moved = True
        for old_pos in old_positions:
            if calculate_distance(new_pos, old_pos) < given_thresh:
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
            if np.linalg.norm(np.array(center) - np.array(unique_center)) < threshold_of_fine_unique_pos:
                is_unique = False
                break
        if is_unique:
            unique_centers.append(center)
    return unique_centers

class SettingsWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Settings")
        self.setGeometry(500, 200, 400, 535)

        QLabel("Slow Scan Time:", self).setGeometry(20, 20, 150, 30)
        self.scan_time_slow_input = QLineEdit(self)
        self.scan_time_slow_input.setGeometry(260, 20, 100, 30)
        self.scan_time_slow_input.setText(str(self.parent.scan_time_slow))

        QLabel("Fast Scan Time:", self).setGeometry(20, 60, 150, 30)
        self.scan_time_fast_input = QLineEdit(self)
        self.scan_time_fast_input.setGeometry(260, 60, 100, 30)
        self.scan_time_fast_input.setText(str(self.parent.scan_time_fast))

        QLabel("Number of Fast Scans:", self).setGeometry(20, 100, 150, 30)
        self.number_of_fast_scan_input = QLineEdit(self)
        self.number_of_fast_scan_input.setGeometry(260, 100, 100, 30)
        self.number_of_fast_scan_input.setText(str(self.parent.number_of_fast_scan))

        QLabel("Flag Limit:", self).setGeometry(20, 140, 150, 30)
        self.flag_limit_input = QLineEdit(self)
        self.flag_limit_input.setGeometry(260, 140, 100, 30)
        self.flag_limit_input.setText(str(self.parent.flag_for_find_true_hexagons_limit))

        QLabel("Missing Time: ", self).setGeometry(20, 180, 200, 30)
        self.missing_time_input = QLineEdit(self)
        self.missing_time_input.setGeometry(260, 180, 100, 30)
        self.missing_time_input.setText(str(self.parent.missing_hexagon_threshold))

        QLabel("Movement Sensitivity:", self).setGeometry(20, 220, 200, 30)
        self.threshold_of_movement_sensitivity_input = QLineEdit(self)
        self.threshold_of_movement_sensitivity_input.setGeometry(260, 220, 100, 30)
        self.threshold_of_movement_sensitivity_input.setText(str(self.parent.threshold_of_movement_sensitivity))

        QLabel("Wind Sensitivity:", self).setGeometry(20, 260, 200, 30)
        self.threshold_of_wind_sensitivity_input = QLineEdit(self)
        self.threshold_of_wind_sensitivity_input.setGeometry(260, 260, 100, 30)
        self.threshold_of_wind_sensitivity_input.setText(str(self.parent.threshold_of_wind_sensitivity))

        QLabel("Red Sensitivity:", self).setGeometry(20, 300, 200, 30)
        self.threshold_of_red_sensitivity_input = QLineEdit(self)
        self.threshold_of_red_sensitivity_input.setGeometry(260, 300, 100, 30)
        self.threshold_of_red_sensitivity_input.setText(str(self.parent.threshold_of_red_sensitivity))

        QLabel("Number of Hexagons:", self).setGeometry(20, 340, 200, 30)
        self.number_of_hexagons_input = QLineEdit(self)
        self.number_of_hexagons_input.setGeometry(260, 340, 100, 30)
        self.number_of_hexagons_input.setText(str(self.parent.number_of_hexagons))

        self.epsilon_label = QLabel(f'Epsilon: {self.parent.epsilon_value:.3f}', self)
        self.epsilon_label.setGeometry(80, 400, 100, 40)

        self.epsilon_slider = QSlider(Qt.Horizontal, self)
        self.epsilon_slider.setEnabled(True)
        self.epsilon_slider.setGeometry(190, 400, 100, 40)
        self.epsilon_slider.setMinimum(1)
        self.epsilon_slider.setMaximum(100)
        self.epsilon_slider.setValue(int(self.parent.epsilon_value * 1000))
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.setGeometry(260, 470, 100, 40)
        self.ok_button.clicked.connect(self.apply_settings)

        self.Reset_button = QPushButton("Reset", self)
        self.Reset_button.setGeometry(150, 470, 100, 40)
        self.Reset_button.clicked.connect(self.reset_settings)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setGeometry(40, 470, 100, 40)
        self.cancel_button.clicked.connect(self.cancel_settings)

    def update_epsilon(self, value):
        epsilon_value = value / 1000.0
        self.epsilon_label.setText(f'Epsilon: {epsilon_value:.3f}')

    def cancel_settings(self):
        self.close()

    def reset_settings(self):
            self.scan_time_slow_input.setText(str(self.parent.default_scan_time_slow))
            self.scan_time_fast_input.setText(str(self.parent.default_scan_time_fast))
            self.number_of_fast_scan_input.setText(str(self.parent.default_number_of_fast_scan))
            self.flag_limit_input.setText(str(self.parent.default_flag_for_find_true_hexagons_limit))
            self.missing_time_input.setText(str(self.parent.default_missing_hexagon_threshold))##
            self.threshold_of_movement_sensitivity_input.setText(str(self.parent.default_threshold_of_movement_sensitivity))
            self.threshold_of_wind_sensitivity_input.setText(str(self.parent.default_threshold_of_wind_sensitivity))
            self.threshold_of_red_sensitivity_input.setText(str(self.parent.default_threshold_of_red_sensitivity))
            self.number_of_hexagons_input.setText(str(self.parent.default_number_of_hexagons))

            self.epsilon_slider.setValue(int(self.parent.default_epsilon_value * 1000))
            self.epsilon_slider.valueChanged.connect(self.update_epsilon)
            
            
    def apply_settings(self):
        self.parent.scan_time_slow = float(self.scan_time_slow_input.text())
        self.parent.scan_time_fast = float(self.scan_time_fast_input.text())
        self.parent.number_of_fast_scan = int(self.number_of_fast_scan_input.text())
        self.parent.flag_for_find_true_hexagons_limit = int(self.flag_limit_input.text())
        self.parent.missing_hexagon_threshold = int(self.missing_time_input.text())
        self.parent.threshold_of_movement_sensitivity = int(self.threshold_of_movement_sensitivity_input.text())
        self.parent.threshold_of_wind_sensitivity = int(self.threshold_of_wind_sensitivity_input.text())
        self.parent.threshold_of_red_sensitivity = int(self.threshold_of_red_sensitivity_input.text())
        self.parent.number_of_hexagons = int(self.number_of_hexagons_input.text())

        self.parent.epsilon_value = self.epsilon_slider.value() / 1000.0
        
        self.close()

class VideoWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shape Detector")
        self.setGeometry(20, 50, 1920, 1080)

        self.video_label = QLabel(self)
        self.video_label.setGeometry(0, 0, 1300, 700)

        self.start_button = QPushButton("Start", self)
        self.start_button.setGeometry(1400, 100, 100, 40)
        self.start_button.clicked.connect(self.start_video)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setGeometry(1400, 160, 100, 40)
        self.stop_button.clicked.connect(self.stop_video)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setGeometry(1400, 220, 100, 40)
        self.exit_button.clicked.connect(self.close)

        self.regions = []

        self.add_button = QPushButton("Add Region", self)
        self.add_button.setGeometry(1400, 280, 100, 40)
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(self.add_region)

        self.delete_button = QPushButton("Delete Region", self)
        self.delete_button.setGeometry(1400, 340, 100, 40)
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_region)

        self.info_box = QTextEdit(self)
        self.info_box.setGeometry(20, 700, 760, 250)
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

        #self.log_df = pd.DataFrame(columns=["timestamp", "region", "hexagons"])

        self.scan_time = 1

        
        ####################################### state
        self.status_label = QLabel("Status:", self)
        self.status_label.setGeometry(1400, 500, 50, 50)
        self.status_label = QLabel(self)
        self.status_label.setGeometry(1450, 500, 50, 50)
        self.update_status("white")

        self.wind_status_label = QLabel("Wind Status:", self)
        self.wind_status_label.setGeometry(1400, 540, 100, 100)
        self.wind_status_label = QLabel(self)
        self.wind_status_label.setGeometry(1490, 575, 30, 30)
        self.wind_status("white")


        self.state = "init"
        self.flag_for_find_true_hexagons = 0
        self.hexagons_last_unique_centers = []
        self.hexagon_true_unique_centers = []
        self.counter_flag = 0
        self.flag_counter_green = 0
        self.flag_counter_red = 0
        self.red_warnings = 0
        self.orange_warnings = 0
        self.wind_warnings = 0
        ##
        self.default_scan_time_slow = 3
        self.default_scan_time_fast = 0.3
        self.default_number_of_fast_scan = 20
        self.default_flag_for_find_true_hexagons_limit = 4
        self.default_epsilon_value = 0.035
        self.default_missing_hexagon_threshold = 5
        self.default_threshold_of_movement_sensitivity = 10
        self.default_threshold_of_wind_sensitivity = 7
        self.default_threshold_of_red_sensitivity = 20
        self.default_number_of_hexagons = 4

        self.scan_time_slow = self.default_scan_time_slow
        self.scan_time_fast = self.default_scan_time_fast
        self.number_of_fast_scan = self.default_number_of_fast_scan
        self.flag_for_find_true_hexagons_limit = self.default_flag_for_find_true_hexagons_limit
        self.epsilon_value = self.default_epsilon_value
        self.missing_hexagon_threshold = self.default_missing_hexagon_threshold
        self.threshold_of_movement_sensitivity = self.default_threshold_of_movement_sensitivity
        self.threshold_of_wind_sensitivity = self.default_threshold_of_wind_sensitivity
        self.threshold_of_red_sensitivity = self.default_threshold_of_red_sensitivity
        self.number_of_hexagons = self.default_number_of_hexagons
        ##

        self.hexagon_missing_timer_start = None 
        self.hexagon_missing_warning_shown = False
        self.ex_state = ""

        self.warning_red_is_enabled = False
        self.fall_warning = 0


        self.warning_label = QLabel("Warnings:", self)
        self.warning_label.setGeometry(1400, 620, 70, 70)

        self.red_warning_lable = QLabel(f"Red: {self.red_warnings}", self)
        self.red_warning_lable.setGeometry(1480, 640, 80, 80)
        self.orange_warning_lable = QLabel(f"Orange: {self.orange_warnings}", self)
        self.orange_warning_lable.setGeometry(1480, 660, 80, 80)
        self.wind_warning_lable = QLabel(f"Wind: {self.wind_warnings}", self)
        self.wind_warning_lable.setGeometry(1480, 680, 80, 80)
        
        self.settings_button = QPushButton("Settings", self)
        self.settings_button.setGeometry(1400, 400, 100, 40)
        self.settings_button.clicked.connect(self.open_settings)

    def open_settings(self):
        self.settings_window = SettingsWindow(self)
        self.settings_window.show()

    def toggle_epsilon_slider(self, state):
        if state == QtCore.Qt.Checked:
            self.epsilon_slider.setEnabled(True)
        else:
            self.epsilon_slider.setEnabled(False)

    def update_status(self, color):
        if color == "green":
            self.status_label.setStyleSheet("background-color: green; border-radius: 25px;")
        elif color == "yellow":
            self.status_label.setStyleSheet("background-color: yellow; border-radius: 25px;")
        elif color == "red":
            self.status_label.setStyleSheet("background-color: red; border-radius: 25px;")
        elif color == "white":
            self.status_label.setStyleSheet("background-color: white; border-radius: 25px;")
        elif color == "orange":
            self.status_label.setStyleSheet("background-color: rgb(255,165,0); border-radius: 25px;")
        elif color == "crimson":
            self.status_label.setStyleSheet("background-color: rgb(153,0,0); border-radius: 25px;")
        elif color == "black":
            self.status_label.setStyleSheet("background-color: rgb(0,0,0); border-radius: 25px;")

    def wind_status(self, color):
        if color == "white":
            self.wind_status_label.setStyleSheet("background-color: white; border-radius: 15;")
        elif color == "yellow":
            self.wind_status_label.setStyleSheet("background-color: yellow; border-radius: 15;")

    # def closeEvent(self, event):
    #     time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     self.log_df.to_csv(f"{time_now}.csv", index=False)
    #     event.accept()

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
                x_ratio = cam_width / self.video_label.width()
                y_ratio = cam_height / self.video_label.height()
                
                self.drawing = True
                self.ix, self.iy = int(event.x() * x_ratio), int(event.y() * y_ratio)
                self.rx1, self.ry1 = self.ix, self.iy
                self.rx2, self.ry2 = self.ix, self.iy
                self.delete_button.setEnabled(False)
                self.update_frame()

    def mouseMoveEvent(self, event):
        if self.is_started:
            if self.drawing:
                x_ratio = cam_width / self.video_label.width()
                y_ratio = cam_height / self.video_label.height()

                self.rx2, self.ry2 = int(event.x() * x_ratio), int(event.y() * y_ratio)
                self.rect_ready = True
                self.update_frame()

    def mouseReleaseEvent(self, event):
        if self.is_started:
            if event.button() == Qt.LeftButton:
                x_ratio = cam_width / self.video_label.width()
                y_ratio = cam_height / self.video_label.height()

                self.drawing = False
                self.rx2, self.ry2 = int(event.x() * x_ratio), int(event.y() * y_ratio)
                self.rect_ready = True
                self.delete_button.setEnabled(True)
                self.update_frame()

    def add_region(self):
        if self.rx1 == self.rx2 or self.ry1 == self.ry2:
            self.info_box.append("No region selected!")
            return

        new_region = (max(0, min(self.rx1, self.rx2)), max(0, min(self.ry1, self.ry2)), min(cam_width, max(self.rx1, self.rx2)), min(cam_height, max(self.ry1, self.ry2)))
        #new_region = (1176,103,1260,163)
 
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
            self.update_status("white")
            self.state = "init"
            self.counter_flag = 0
            self.red_warnings = 0
            self.wind_warnings = 0
            self.orange_warnings = 0
            self.wind_warning_lable.setText(f"Wind: {self.wind_warnings}")
            self.red_warning_lable.setText(f"Red: {self.red_warnings}")
            self.orange_warning_lable.setText(f"Orange: {self.orange_warnings}")
            self.update_frame()
        
        self.rx1, self.ry1, self.rx2, self.ry2 = 0, 0, 0, 0
        self.rect_ready = False

    def check_missing_hexagons(self, current_time, hexagons_unique_centers):
        if len(hexagons_unique_centers) < len(self.hexagon_true_unique_centers) / 2:
            if self.hexagon_missing_timer_start is None:
                self.hexagon_missing_timer_start = current_time
            elif current_time - self.hexagon_missing_timer_start >= self.missing_hexagon_threshold:
                if not self.hexagon_missing_warning_shown:
                    self.info_box.append("Warning: Less than half of the hexagons have been detected.")
                    if not self.warning_red_is_enabled:
                        self.update_status("orange")
                    self.hexagon_missing_warning_shown = True
                    self.info_box.append("State: Orange")
                    self.state = "orange"
                    self.orange_warnings += 1
                    self.orange_warning_lable.setText(f"Orange: {self.orange_warnings}")
        else:
            self.hexagon_missing_timer_start = None
            self.hexagon_missing_warning_shown = False


    def update_frame(self):
        ret, frame = cam.read()
        if ret:
            for region in self.regions:
                x1, x2 = max(0, min(region[0], region[2])), min(cam_width, max(region[0], region[2]))
                y1, y2 = max(0, min(region[1], region[3])), min(cam_height, max(region[1], region[3]))
                roi = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #equalized = cv2.equalizeHist(gray) #For Night
                blurred = cv2.GaussianBlur(gray, (3, 3), 5)
                thresh = cv2.adaptiveThreshold(blurred, 150, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                detected_hexagons = detect_shapes(contours, self.epsilon_value)

                if region not in self.region_last_check_times:
                    self.region_last_check_times[region] = datetime.now()
                if region not in self.region_last_detected_times:
                    self.region_last_detected_times[region] = datetime.now()
                if region not in self.region_last_alarm_times:
                    self.region_last_alarm_times[region] = datetime.now()
                if region not in self.region_hexagon_positions:
                    self.region_hexagon_positions[region] = []

                hexagons_last_centers = []

                for hexagon in detected_hexagons:
                    M = cv2.moments(hexagon)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        hexagons_last_centers.append((cx, cy))
                        cv2.circle(roi, (cx, cy), 3, (0, 0, 255), -1)

                hexagons_unique_centers = count_unique_positions(hexagons_last_centers)

                current_time = time.time()
                if region not in self.region_last_count_time:
                    self.region_last_count_time[region] = current_time

########################################################################################################                    

                if current_time - self.region_last_count_time[region] >= self.scan_time:
                    if current_time - self.region_last_count_time[region] >= self.scan_time:
                        if check_movement(hexagons_unique_centers, self.hexagons_last_unique_centers, self.threshold_of_wind_sensitivity) and not self.state == "fast_scan" and not self.state == "init":
                            self.wind_status("yellow")
                        elif not self.state == "fast_scan":
                            self.wind_status("white")

                    if self.state == "init":
                        if self.counter_flag == 6:
                            # self.rx1 = 0
                            # self.rx2 = cam_width
                            # self.ry1 = 0
                            # self.ry2 = cam_height
                            # self.add_region()
                            self.counter_flag = 0
                        else:
                            self.counter_flag += 1

                        if self.flag_for_find_true_hexagons < self.flag_for_find_true_hexagons_limit:
                            if self.hexagons_last_unique_centers and not check_movement(hexagons_unique_centers, self.hexagons_last_unique_centers, self.threshold_of_movement_sensitivity):
                                self.flag_for_find_true_hexagons += 1
                            else: 
                                self.flag_for_find_true_hexagons = 0
                        else:
                            if len(hexagons_unique_centers) >= self.number_of_hexagons:
                                self.flag_for_find_true_hexagons = 0
                                self.info_box.append("State: Slow Scan")
                                self.state = "slow_scan"
                                self.hexagon_true_unique_centers = hexagons_unique_centers
                                self.info_box.append(f"Number of Hexagons Detected: {len(self.hexagon_true_unique_centers)}")
                                self.update_status("green")

                    elif self.state == "slow_scan":
                        self.ex_state = self.state
                        self.scan_time = self.scan_time_slow
                        if check_movement(hexagons_unique_centers, self.hexagons_last_unique_centers, self.threshold_of_movement_sensitivity) or check_movement(hexagons_unique_centers, self.hexagon_true_unique_centers, self.threshold_of_red_sensitivity):
                            self.info_box.append("State: Fast Scan")
                            self.state = "fast_scan"
                            self.scan_time = self.scan_time_fast
                            self.counter_flag = 0
                        self.check_missing_hexagons(current_time, hexagons_unique_centers)


                    elif self.state == "fast_scan":
                        self.ex_state = self.state
                        self.scan_time = self.scan_time_fast
                        if not self.counter_flag == self.number_of_fast_scan:
                            if check_movement(hexagons_unique_centers, self.hexagon_true_unique_centers, self.threshold_of_red_sensitivity):
                                self.flag_counter_red += 1
                            else:
                                self.flag_counter_green += 1
                            self.counter_flag += 1
                        else:
                            self.state = "decision"
                            self.counter_flag = 0
                            self.info_box.append("State: Decision")

                        self.check_missing_hexagons(current_time, hexagons_unique_centers)

                    elif self.state == "decision":
                        if max(self.flag_counter_green, self.flag_counter_red) == self.flag_counter_red:
                            self.flag_counter_red = 0
                            self.flag_counter_green = 0
                            self.update_status("red")
                            self.state = "red_calibration"
                            self.red_warnings += 1
                            self.red_warning_lable.setText(f"Red: {self.red_warnings}")
                            self.info_box.append("Warning: Hexagons moved.")
                            self.info_box.append("State: Red")

                        elif max(self.flag_counter_green, self.flag_counter_red) == self.flag_counter_green:
                            self.flag_counter_red = 0
                            self.flag_counter_green = 0
                            self.state = "slow_scan"
                            if not self.warning_red_is_enabled:
                                self.update_status("green")
                            self.info_box.append("State: Slow Scan")

                    elif self.state == "orange":
                        self.scan_time = self.scan_time_slow
                        if len(hexagons_unique_centers) == 0:
                            if self.fall_warning == 2:
                                self.state = "black"
                            else:
                                self.fall_warning += 1
                        elif len(hexagons_unique_centers) < len(self.hexagon_true_unique_centers) / 2:
                            self.fall_warning = 0
                        else:
                            self.fall_warning = 0
                            self.hexagon_missing_timer_start = None
                            self.hexagon_missing_warning_shown = False
                            self.state = self.ex_state

                            if self.state == "slow_scan":
                                if not self.warning_red_is_enabled:
                                    self.update_status("green")
                                self.info_box.append("State: Slow Scan")

                    elif self.state == "black":
                        self.info_box.append("!!! Fall Warning !!!")
                        #cv2.putText(frame, "!!! Fall Warning !!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        self.update_status("black")
                    
                    elif self.state == "red_calibration":
                        self.warning_red_is_enabled = True
                        self.info_box.append("State: Slow Scan")
                        self.state = "slow_scan"
                        if len(hexagons_unique_centers) == len(self.hexagon_true_unique_centers):
                            self.hexagon_true_unique_centers = hexagons_unique_centers

                    self.region_last_count_time[region] = current_time
                    self.hexagons_last_unique_centers = hexagons_unique_centers

##################################################################################################

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"({x1}, {y1}, {x2}, {y2})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        
            if self.drawing or (self.rx1 != 0 and self.ry1 != 0 and self.rx2 != 0 and self.ry2 != 0):
                cv2.rectangle(frame, (self.rx1, self.ry1), (self.rx2, self.ry2), (0, 255, 0), 2)
                text = f"({self.rx1}, {self.ry1}, {self.rx2}, {self.ry2})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text, (self.rx1, self.ry1 - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            qformat = QImage.Format_RGB888
            frame = cv2.resize(frame, (1300, 700))
            out_image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
            out_image = out_image.rgbSwapped()
            self.video_label.setPixmap(QPixmap.fromImage(out_image))

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
