#-----------------------------------------------------------
# Title: Telecommunication Tower Stability Analysis Using Computer Vision
# Author: Amirhossein Abbasifar
# Email: amirhosseinabbasifar@gmail.com
# Company: Nader Niroo Gharb Razi
# Date: Bahman 17, 1403
# Version: 1.0
#-----------------------------------------------------------
#
# Description:
# This project aims at analyzing the stability of telecommunication towers to identify potential structural problems and instabilities. 
# Utilizing advanced computer vision techniques in conjunction with various sensors, the system provides a comprehensive analysis 
# of the tower's state. The methodology includes several key stages:
#
# 1. Problem Analysis and Objectives: Assessing the structural integrity of telecommunication towers to identify and flag 
#    issues related to stability. This involves the integration of computer vision and data from multiple sensors for effective monitoring.
#
# 2. Data Collection: Gathering data through the use of cameras and supplementary sensors such as:
#    - Gyroscope sensors for measuring orientation changes.
#    - Accelerometers for detecting movement and vibrations.
#    - Anemometers for wind speed measurement.
#    - Pressure or force sensors to detect external stresses on the tower.
#    This multisensory data, combined with visual data, allows for a more accurate assessment of the tower's condition.
#
# 3. Image Processing and Feature Detection: 
#    - Implementing preprocessing techniques to enhance image quality (i.e., noise reduction, contrast enhancement).
#    - Identifying and extracting essential features from images, such as edges and lines.
#    - Conducting comparative analysis of images taken at different time intervals to identify any changes or trends.
#
# 4. Data Integration: By merging data from various sensors with images, a more detailed assessment of tower conditions can be achieved. 
#    For instance, correlating angle changes detected by the gyroscope with wind sensor data and laser leveling readings can indicate 
#    structural instabilities.
#
# Output:
# The outcome of this project is a sophisticated system designed to analyze the stability of telecommunication towers. 
# This software and hardware solution is capable of interpreting both visual and sensor data, effectively identifying 
# structural issues and potential instabilities. By utilizing real-time data and advanced analytics, the system aims to enhance 
# the safety and reliability of telecommunication infrastructure.
#-----------------------------------------------------------

# Import necessary libraries
import cv2
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QElapsedTimer
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget, QApplication, QTextEdit, QSlider, QSpinBox, QLineEdit
from PyQt5.QtGui import QImage, QPixmap
import os
import atexit
import requests
import threading
import tkinter as tk
from tkinter import messagebox

# Debug Mode
debug = True

# Global variables
wind_speed = 0
threshold_of_fine_unique_pos = 30
area_limit = 80
frame_rate_threshold = 33  # 33 ms for ~30 FPS - 1000 ms for 1 FPS

# Coordinates for the specific location
LATITUDE = 34.252016197565254  # NirooGharb
LONGITUDE = 47.02910396696085  # NirooGharb

# API endpoint for fetching weather data
url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&current_weather=true"

# Camera URL
camera_url = 'rtsp://admin:admin1234@91.92.231.159:39735/cam/realmonitor?channel=1&subtype=0'
#camera_url = 'rtsp://admin:admin1234@172.16.90.232:554/cam/realmonitor?channel=1&subtype=0'

def setup_logging():
    """
    Sets up the logging directory and returns the log file path.
    
    Returns:
        log_file_path: The path to the log file.
    """

    logs_dir = os.path.join(os.getcwd(), "Logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    today_dir = os.path.join(logs_dir, datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(today_dir):
        os.makedirs(today_dir)

    log_file_path = os.path.join(today_dir, f"hexagon_vertices_{datetime.now().strftime('%H-%M-%S')}.csv")
    return log_file_path

def show_error(message):
    """Displays an error message in a Tkinter window."""
    root = tk.Tk()
    root.withdraw()
    ErrorWindow = tk.Toplevel(root)
    
    ErrorWindow.title("Camera Connection Error")
    ErrorWindow.geometry("300x150")

    label = tk.Label(ErrorWindow, text=message, wraplength=250)
    label.pack(pady=20)

    ok_button = tk.Button(ErrorWindow, text="OK", command=root.destroy)
    ok_button.pack(pady=10)

    root.mainloop()

def enhance_image(roi, debug):
    """
    Enhances the ROI (Region of Interest) for better processing and detection.
    
    Args:
        roi (numpy.ndarray): The region of interest from the frame.
        debug (bool): If True, shows intermediate images for debugging.
    
    Returns:
        numpy.ndarray: The enhanced and thresholded image ready for further processing.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imshow("Gray", gray)

    # Apply gamma correction
    gamma = 1.5  # Gamma correction factor
    lookup_table = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, lookup_table)
    if debug:
        cv2.imshow("Gamma Corrected", gamma_corrected)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gamma_corrected, (7, 7), 1.5)
    if debug:
        cv2.imshow("Blurred Gaussian", blurred)

    # Apply median blur
    blurred_median = cv2.medianBlur(blurred, 5)
    if debug:
        cv2.imshow("Blurred Median", blurred_median)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    equalized = clahe.apply(blurred_median)
    if debug:
        cv2.imshow("CLAHE", equalized)

    # Apply histogram equalization
    # equalized2 = cv2.equalizeHist(equalized)
    # if debug:
    #     cv2.imshow("Equalized", equalized2)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    if debug:
        cv2.imshow("Threshold", thresh)

    return thresh

def is_approximate_hexagon(approx, min_side_length=11, max_side_length=1000):
    """Checks if the given contour is an approximate hexagon.
    
    Args:
        approx: ApproxPolyDP of the contour.
        min_side_length: Minimum length of each side.
        max_side_length: Maximum length of each side.
    
    Returns:
        True if the contour is an approximate hexagon.
    """
    if len(approx) != 6:
        return False
    if not cv2.isContourConvex(approx):
        return False

    side_lengths = []
    angles = []

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
        if angle < 60 or angle > 170:
            return False

    side_lengths = np.array(side_lengths)
    mean_length = np.mean(side_lengths)
    if np.any(side_lengths < min_side_length) or np.any(side_lengths > max_side_length):
        return False

    length_deviation = np.std(side_lengths)
    if length_deviation > 0.25 * mean_length:
        return False

    return True

def update_wind_speed():
    """Continuously updates the wind speed every 60 seconds."""
    global wind_speed
    while True:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            wind_speed = data["current_weather"]["windspeed"]
        except Exception as e:
            pass
        time.sleep(60)

def detect_shapes(contours, epsilon_val, log_file_path):
    """Detects hexagons from the list of contours.
    
    Args:
        contours: List of contours.
        epsilon_val: Epsilon value for contour approximation.
        log_file_path: Path of log
    
    Returns:
        List of detected hexagons.
    """
    hexagons = []
    new_data = []

    hexagon_centers = []

    for contour in contours:
        epsilon = epsilon_val * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if is_approximate_hexagon(approx):
            area = cv2.contourArea(contour)
            if area > area_limit:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hexagon_centers.append((cx, cy))
                    hexagons.append(contour)

    unique_centers = count_unique_positions(hexagon_centers)

    unique_hexagons = []
    for i, center in enumerate(hexagon_centers):
        if center in unique_centers:
            unique_hexagons.append(hexagons[i])

    for i, center in enumerate(unique_centers):
        contour = unique_hexagons[i]
        approx = cv2.approxPolyDP(contour, epsilon_val * cv2.arcLength(contour, True), True)
        vertices = approx[:, 0, :]
        
        # hexagon_data = {
        #     'x1': f"{vertices[0][0]}",
        #     'y1': f"{vertices[0][1]}",
        #     'x2': f"{vertices[1][0]}",
        #     'y2': f"{vertices[1][1]}",
        #     'x3': f"{vertices[2][0]}",
        #     'y3': f"{vertices[2][1]}",
        #     'x4': f"{vertices[3][0]}",
        #     'y4': f"{vertices[3][1]}",
        #     'x5': f"{vertices[4][0]}",
        #     'y5': f"{vertices[4][1]}",
        #     'x6': f"{vertices[5][0]}",
        #     'y6': f"{vertices[5][1]}",
        #     'Center_x': f"{center[0]}",
        #     'Center_y': f"{center[1]}",
        #     'Wind Speed': f"{wind_speed} km/h"
        # }
        lowest_x = vertices[0][0]
        lowest_x_y = vertices[0][1]

        highest_x = vertices[0][0]
        highest_x_y = vertices[0][1]

        lowest_y = vertices[0][1]
        lowest_y_x = vertices[0][0]

        highest_y = vertices[0][1]
        highest_y_x = vertices[0][0]
        

        for i in range(0,6):
            if vertices[i][0] < lowest_x: #find lowest x and its y
                lowest_x = vertices[i][0]
                lowest_x_y = vertices[i][1]

            if vertices[i][0] > highest_x: #find highest x and its y
                highest_x = vertices[i][0]
                highest_x_y = vertices[i][1]

            if vertices[i][1] < lowest_y: #find lowest y and its x
                lowest_y = vertices[i][1]
                lowest_y_x = vertices[i][0]
            
            if vertices[i][1] > highest_y: #find highest y and its x
                highest_y = vertices[i][1]
                highest_y_x = vertices[i][0]

        mid_up_y = vertices[0][0]
        mid_up_y_x = vertices[0][1]

        mid_down_y = vertices[0][0]
        mid_down_y_x = vertices[0][1]
 
        for i in range (0,6):
            if vertices[i][1] != highest_y and vertices[i][1] != lowest_y and vertices[i][1] != lowest_x_y and vertices[i][1] != highest_x_y:
                if vertices[i][1] > mid_up_y:
                    mid_up_y = vertices[i][1]
                    mid_up_y_x = vertices[i][0]
                if vertices[i][1] < mid_down_y:
                    mid_down_y = vertices[i][1]
                    mid_up_y_x = vertices[i][0]

        hexagon_data = {
            'x1': f"{lowest_x}",
            'y1': f"{lowest_x_y}",
            'x2': f"{highest_y_x}",
            'y2': f"{highest_y}",
            'x3': f"{mid_up_y_x}",
            'y3': f"{mid_up_y}",
            'x4': f"{highest_x}",
            'y4': f"{highest_x_y}",
            'x5': f"{lowest_y_x}",
            'y5': f"{lowest_y}",
            'x6': f"{mid_down_y_x}",
            'y6': f"{mid_down_y}",
            'Center_x': f"{center[0]}",
            'Center_y': f"{center[1]}",
            'Wind Speed': f"{wind_speed} km/h"
        }
        new_data.append(hexagon_data)

    if new_data:
        new_df = pd.DataFrame(new_data)
        new_df.to_csv(log_file_path, mode='a', header=not os.path.exists(log_file_path), index=False)

    return unique_hexagons

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points.
    
    Args:
        p1: First point (x, y).
        p2: Second point (x, y).
        
    Returns:
        Euclidean distance between p1 and p2.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def check_movement(new_positions, old_positions, given_thresh):
    """Checks if any of the new positions have moved significantly from the old positions.
    
    Args:
        new_positions: List of new positions.
        old_positions: List of old positions.
        given_thresh: Threshold distance for movement detection.
        
    Returns:
        True if any position has moved significantly.
    """
    if not new_positions or not old_positions:
        return False
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
    """Counts the number of unique positions within a given threshold distance.
    
    Args:
        centers: List of positions (x, y).
        
    Returns:
        List of unique positions.
    """
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

class CameraConnector(threading.Thread):
    """Thread to handle camera connection and initialization."""
    def __init__(self, camera_url):
        super().__init__()
        self.camera_url = camera_url  # URL دوربین
        self.cam = None
        self.cam_width = None
        self.cam_height = None

    def run(self):
        """Attempts to open the camera connection using the provided URL."""
        self.cam = cv2.VideoCapture(self.camera_url)
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID'))
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Disable buffering
        self.cam.set(cv2.CAP_PROP_POS_MSEC, 3000)  # Set timeout to 3 seconds

    def initialize(self):
        """
        Initializes the camera connection and retrieves its properties.
        
        Returns:
            cam: The camera object.
            cam_width: Width of the camera frame.
            cam_height: Height of the camera frame.
        """
        self.start()
        self.join(timeout=10)  # Wait for 10 seconds

        if self.cam is None or not self.cam.isOpened():
            show_error("Failed to connect to camera")
            os._exit(1)
        else:
            try:
                self.cam_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.cam_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return self.cam, self.cam_width, self.cam_height
            except Exception as e:
                print(f"Error getting camera properties: {e}")
                self.cam.release()
                os._exit(1)

class VideoThread(QThread):
    """Thread to handle video frame fetching."""
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cam = None
        self.connect_camera()

    def connect_camera(self):
        """Attempts to open the camera connection."""
        self.cam = cv2.VideoCapture('rtsp://admin:admin1234@91.92.231.159:39735/cam/realmonitor?channel=1&subtype=0')

    def run(self):
        """Runs the video frame fetching loop."""
        while self._run_flag:
            if self.cam is None or not self.cam.isOpened():
                self.connect_camera()
                time.sleep(1)  # Wait before retrying

            ret, frame = self.cam.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
            else:
                self.cam.release()
                self.cam = None
            time.sleep(0.033)  # 30 frames per second

    def stop(self):
        """Stops the video frame fetching thread."""
        self._run_flag = False
        if self.cam is not None:
            self.cam.release()
        self.wait()

class SettingsWindow(QWidget):
    """Window for changing application settings."""
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

        QLabel("Missing Time:", self).setGeometry(20, 180, 200, 30)
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

        self.reset_button = QPushButton("Reset", self)
        self.reset_button.setGeometry(150, 470, 100, 40)
        self.reset_button.clicked.connect(self.reset_settings)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setGeometry(40, 470, 100, 40)
        self.cancel_button.clicked.connect(self.cancel_settings)

    def update_epsilon(self, value):
        """Updates the epsilon value based on slider change."""
        epsilon_value = value / 1000.0
        self.epsilon_label.setText(f'Epsilon: {epsilon_value:.3f}')

    def cancel_settings(self):
        """Closes the settings window."""
        self.close()

    def reset_settings(self):
        """Resets the settings to default values."""
        self.scan_time_slow_input.setText(str(self.parent.default_scan_time_slow))
        self.scan_time_fast_input.setText(str(self.parent.default_scan_time_fast))
        self.number_of_fast_scan_input.setText(str(self.parent.default_number_of_fast_scan))
        self.flag_limit_input.setText(str(self.parent.default_flag_for_find_true_hexagons_limit))
        self.missing_time_input.setText(str(self.parent.default_missing_hexagon_threshold))
        self.threshold_of_movement_sensitivity_input.setText(str(self.parent.default_threshold_of_movement_sensitivity))
        self.threshold_of_wind_sensitivity_input.setText(str(self.parent.default_threshold_of_wind_sensitivity))
        self.threshold_of_red_sensitivity_input.setText(str(self.parent.default_threshold_of_red_sensitivity))
        self.number_of_hexagons_input.setText(str(self.parent.default_number_of_hexagons))

        self.epsilon_slider.setValue(int(self.parent.default_epsilon_value * 1000))
        self.epsilon_slider.valueChanged.connect(self.update_epsilon)

    def apply_settings(self):
        """Applies the new settings from the input fields."""
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
    """Main application window for video processing and interaction."""
    
    def __init__(self, cam, cam_width, cam_height, log_file_path):
        super().__init__()
        self.cam = cam
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.log_file_path = log_file_path

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

        self.add_region_button = QPushButton("Add Region", self)
        self.add_region_button.setGeometry(1400, 280, 100, 40)
        self.add_region_button.setEnabled(False)
        self.add_region_button.clicked.connect(self.add_region)

        self.delete_region_button = QPushButton("Delete Region", self)
        self.delete_region_button.setGeometry(1400, 340, 100, 40)
        self.delete_region_button.setEnabled(False)
        self.delete_region_button.clicked.connect(self.delete_region)

        self.info_box = QTextEdit(self)
        self.info_box.setGeometry(20, 700, 760, 250)
        self.info_box.setReadOnly(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.drawing = False
        self.ix, self.iy = -1, -1
        self.rx1, self.ry1, self.rx2, self.ry2 = 0, 0, 0, 0

        self.rect_ready = False
        self.is_started = False

        self.region_last_count_time = {}
        self.scan_time = 1

        self.elapsed_thresh = frame_rate_threshold

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

        self.wind_speed_label = QLabel(f"Wind Speed: {wind_speed} km/h", self)
        self.wind_speed_label.setGeometry(1400, 585, 150, 80)

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

        self.default_scan_time_slow = 3
        self.default_scan_time_fast = 0.3
        self.default_number_of_fast_scan = 20
        self.default_flag_for_find_true_hexagons_limit = 4
        self.default_epsilon_value = 0.032
        self.default_missing_hexagon_threshold = 30
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

        self.hexagon_missing_timer_start = None
        self.hexagon_missing_warning_shown = False
        self.ex_state = ""

        self.warning_red_is_enabled = False
        self.fall_warning = 0

        self.timer = QElapsedTimer()
        self.timer.start()

        self.start_duration_time = None
        self.log_file_path = log_file_path
        atexit.register(self.log_duration)

        self.regions = []

        self.warning_label = QLabel("Warnings:", self)
        self.warning_label.setGeometry(1400, 620, 70, 70)

        self.red_warning_label = QLabel(f"Moved: {self.red_warnings}", self)
        self.red_warning_label.setGeometry(1480, 640, 80, 80)
        self.orange_warning_label = QLabel(f"Not detected: {self.orange_warnings}", self)
        self.orange_warning_label.setGeometry(1480, 660, 120, 80)
        self.wind_warning_label = QLabel(f"Wind: {self.wind_warnings}", self)
        self.wind_warning_label.setGeometry(1480, 680, 80, 80)

        self.settings_button = QPushButton("Settings", self)
        self.settings_button.setGeometry(1400, 400, 100, 40)
        self.settings_button.clicked.connect(self.open_settings)

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_frame)

        self.wind_speed_thread = threading.Thread(target=update_wind_speed, daemon=True)
        self.wind_speed_thread.start()

    def open_settings(self):
        """Opens the settings window."""
        self.settings_window = SettingsWindow(self)
        self.settings_window.show()

    def update_status(self, color):
        """Updates the status label color based on the current state.
        
        Args:
            color: Color to set the status label.
        """
        if color == "green":
            self.status_label.setStyleSheet("background-color: green; border-radius: 25px;")
        elif color == "red":
            self.status_label.setStyleSheet("background-color: red; border-radius: 25px;")
        elif color == "white":
            self.status_label.setStyleSheet("background-color: white; border-radius: 25px;")
        elif color == "orange":
            self.status_label.setStyleSheet("background-color: rgb(255,165,0); border-radius: 25px;")
        elif color == "black":
            self.status_label.setStyleSheet("background-color: rgb(0,0,0); border-radius: 25px;")

    def wind_status(self, color):
        """Updates the wind status label color based on the current wind status.
        
        Args:
            color: Color to set the wind status label.
        """
        if color == "white":
            self.wind_status_label.setStyleSheet("background-color: white; border-radius: 15px;")
        elif color == "yellow":
            self.wind_status_label.setStyleSheet("background-color: yellow; border-radius: 15px;")

    def start_video(self):
        """Starts the video processing thread."""
        self.is_started = True
        self.video_thread.start()
        self.add_region_button.setEnabled(True)
        self.delete_region_button.setEnabled(True)

    def stop_video(self):
        """Stops the video processing thread and logs the duration."""
        self.log_duration()
        self.log_file_path = setup_logging()
        self.start_duration_time = time.time()
        self.add_region_button.setEnabled(False)
        self.delete_region_button.setEnabled(False)

    def log_duration(self):
        """Logs the duration of the video processing session."""
        if self.start_duration_time is not None:
            end_time = time.time()  # End time of the session
            duration = int(end_time - self.start_duration_time)  # Duration in seconds

            # Calculate FPS (Frames per Second)
            fps = int(1000 / self.elapsed_thresh)

            # Read existing CSV file or create a new one
            if os.path.exists(self.log_file_path):
                df = pd.read_csv(self.log_file_path)
            else:
                df = pd.DataFrame()

            # Add each column separately
            df.loc[0, "Start Time"] = datetime.fromtimestamp(self.start_duration_time).strftime("%Y-%m-%d_%H:%M")
            df.loc[0, "Stop Time"] = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d_%H:%M")
            df.loc[0, "Duration"] = f"{duration}s"
            df.loc[0, "FPS"] = fps
            df.loc[0, "Moved Warning"] = self.red_warnings
            df.loc[0, "Not detected Warning"] = self.orange_warnings
            df.loc[0, "Wind Warning"] = self.wind_warnings

            # Save the CSV file
            df.to_csv(self.log_file_path, index=False)

    def mousePressEvent(self, event):
        """Handles mouse press event for drawing regions.
        
        Args:
            event: Mouse event.
        """
        if self.is_started:
            if event.button() == Qt.LeftButton:
                x_ratio = self.cam_width / self.video_label.width()
                y_ratio = self.cam_height / self.video_label.height()

                self.drawing = True
                self.ix, self.iy = int(event.x() * x_ratio), int(event.y() * y_ratio)
                self.rx1, self.ry1 = self.ix, self.iy
                self.rx2, self.ry2 = self.ix, self.iy
                self.delete_region_button.setEnabled(False)

    def mouseMoveEvent(self, event):
        """Handles mouse move event for drawing regions.
        
        Args:
            event: Mouse event.
        """
        if self.is_started:
            if self.drawing:
                x_ratio = self.cam_width / self.video_label.width()
                y_ratio = self.cam_height / self.video_label.height()

                self.rx2, self.ry2 = int(event.x() * x_ratio), int(event.y() * y_ratio)
                self.rect_ready = True

    def mouseReleaseEvent(self, event):
        """Handles mouse release event for drawing regions.
        
        Args:
            event: Mouse event.
        """
        if self.is_started:
            if event.button() == Qt.LeftButton:
                x_ratio = self.cam_width / self.video_label.width()
                y_ratio = self.cam_height / self.video_label.height()

                self.drawing = False
                self.rx2, self.ry2 = int(event.x() * x_ratio), int(event.y() * y_ratio)
                self.rect_ready = True
                self.delete_region_button.setEnabled(True)

    def add_region(self):
        """Adds a region of interest based on the drawn rectangle."""
        self.start_duration_time = time.time()
        if self.rx1 == self.rx2 or self.ry1 == self.ry2:
            self.info_box.append("No region selected!")
            return

        new_region = (max(0, min(self.rx1, self.rx2)), max(0, min(self.ry1, self.ry2)), min(self.cam_width, max(self.rx1, self.rx2)), min(self.cam_height, max(self.ry1, self.ry2)))

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

    def delete_region(self):
        """Deletes the last added region of interest."""
        if self.regions:
            self.regions.pop()
            self.info_box.append("Last region deleted.")
            self.add_region_button.setEnabled(True)
            self.rect_ready = False
            self.update_status("white")
            self.state = "init"
            self.counter_flag = 0
            self.red_warnings = 0
            self.wind_warnings = 0
            self.orange_warnings = 0
            self.wind_warning_label.setText(f"Wind: {self.wind_warnings}")
            self.red_warning_label.setText(f"Moved: {self.red_warnings}")
            self.orange_warning_label.setText(f"Not detected: {self.orange_warnings}")

        self.rx1, self.ry1, self.rx2, self.ry2 = 0, 0, 0, 0
        self.rect_ready = False

    def check_missing_hexagons(self, current_time, hexagons_unique_centers):
        """Checks if the number of detected hexagons is less than half of the expected number.
        
        Args:
            current_time: Current time.
            hexagons_unique_centers: List of unique hexagon centers.
        """
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
                    self.orange_warning_label.setText(f"Not detected: {self.orange_warnings}")
        else:
            self.hexagon_missing_timer_start = None
            self.hexagon_missing_warning_shown = False

    def update_frame(self, frame):
        """Updates the video frame and performs hexagon detection and state management.
        
        Args:
            frame: Current video frame.
        """
        elapsed = self.timer.elapsed()
        if elapsed >= self.elapsed_thresh:

            self.wind_speed_label.setText(f"Wind Speed: {wind_speed} km/h")

            if not self.is_started:
                return

            if self.drawing or (self.rx1 != 0 and self.ry1 != 0 and self.rx2 != 0 and self.ry2 != 0):
                cv2.rectangle(frame, (self.rx1, self.ry1), (self.rx2, self.ry2), (0, 255, 0), 2)
                text = f"({self.rx1}, {self.ry1}, {self.rx2}, {self.ry2})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text, (self.rx1, self.ry1 - 10), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            for region in self.regions:
                x1, x2 = max(0, min(region[0], region[2])), min(self.cam_width, max(region[0], region[2]))
                y1, y2 = max(0, min(region[1], region[3])), min(self.cam_height, max(region[1], region[3]))
                roi = frame[y1:y2, x1:x2]

                enhanced_image = enhance_image(roi, debug)

                contours, hierarchy = cv2.findContours(enhanced_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

                detected_hexagons = detect_shapes(contours, self.epsilon_value, self.log_file_path)
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

                if current_time - self.region_last_count_time[region] >= self.scan_time:
                    if current_time - self.region_last_count_time[region] >= self.scan_time:
                        if check_movement(hexagons_unique_centers, self.hexagons_last_unique_centers, self.threshold_of_wind_sensitivity) and not self.state == "fast_scan" and not self.state == "init":
                            self.wind_status("yellow")
                            self.wind_warnings += 1
                            self.wind_warning_label.setText(f"Wind: {self.wind_warnings}")
                        elif not self.state == "fast_scan":
                            self.wind_status("white")

                    if self.state == "init":
                        if self.counter_flag == 6:
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
                            self.red_warning_label.setText(f"Moved: {self.red_warnings}")
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
                        self.update_status("black")

                    elif self.state == "red_calibration":
                        self.warning_red_is_enabled = True
                        self.info_box.append("State: Slow Scan")
                        self.state = "slow_scan"
                        if len(hexagons_unique_centers) == len(self.hexagon_true_unique_centers):
                            self.hexagon_true_unique_centers = hexagons_unique_centers

                    self.region_last_count_time[region] = current_time
                    self.hexagons_last_unique_centers = hexagons_unique_centers

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"({x1}, {y1}, {x2}, {y2})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            qformat = QImage.Format_RGB888
            frame = cv2.resize(frame, (1300, 700))
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            out_image = QImage(rgb_image.data, w, h, ch * w, qformat)
            self.video_label.setPixmap(QPixmap.fromImage(out_image))
            self.timer.restart()

if __name__ == '__main__':
    import sys
    
    # Initialize camera
    connector = CameraConnector(camera_url)
    cam, cam_width, cam_height = connector.initialize()

    # Set up logging
    log_file_path = setup_logging()

    # Start the main application
    app = QApplication(sys.argv)
    window = VideoWindow(cam, cam_width, cam_height, log_file_path)
    window.show()
    sys.exit(app.exec_())