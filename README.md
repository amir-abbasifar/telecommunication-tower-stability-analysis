# Hexagon Shape Detector with Wind Speed Monitoring

## Author

Amirhossein Abbasifar

Email: [amirhosseinabbasifar@gmail.com](mailto:amirhosseinabbasifar@gmail.com)

Company: Nader Niroo Gharb Razi

---

## Description

This application is designed to monitor a live video feed for hexagonal shapes and log their positions along with wind speed data. It employs OpenCV for image processing and PyQt5 for the graphical user interface. The application can detect hexagons, track their movements, and log warnings if there is significant movement or fewer hexagons than expected.

## Features

- **Hexagon Detection**: Utilizes contour detection and approximation to identify hexagons.
- **Wind Speed Monitoring**: Fetches wind speed data from an online API.
- **Region of Interest (ROI)**: Users can define regions to track hexagons.
- **State Management**: Manages various states such as slow scan, fast scan, decision, and calibration.
- **Logging**: Records critical data including positions, timestamps, and warnings.

## Usage

1. Ensure all dependencies are installed:
   - OpenCV
   - PyQt5
   - pandas
   - requests
   - numpy
2. Run the script to start video processing and monitoring.

## Dependencies

You can install the required packages using pip:

```bash
pip install opencv-python PyQt5 pandas requests numpy
```

## Clone the Repository

To clone this repository, you can use the following command:

```bash
git clone https://github.com/Amirabs0/Hexagon_Detection
```
