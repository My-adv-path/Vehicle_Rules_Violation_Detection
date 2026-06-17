# Vehicle_Rules_Violation_Detection


### 🚦 Overview

This project is a **Computer Vision–based Vehicle Rules Violation Detection System** that automatically detects vehicles violating traffic signals, tracks their movement, recognizes license plates, and stores violation evidence. The system leverages **YOLOv8**, **OpenCV**, **EasyOCR**, and object tracking to perform real-time traffic monitoring from video streams.

### ✨ Features

* Real-time vehicle detection using YOLOv8
* Multi-object vehicle tracking
* Traffic signal state monitoring (Red/Green)
* Automatic red-light violation detection
* License plate detection and extraction
* Optical Character Recognition (OCR) using EasyOCR
* Violation image capture and storage
* Automatic saving of detected license plate images
* Supports cars, buses, trucks, motorcycles, and bicycles

### 🛠️ Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* EasyOCR
* NumPy
* Pandas

### 📂 Project Structure

Vehicle-Rules-Violation-Detection/
│
├── violation/              # Stores captured violation images
├── plates/                 # Stores detected license plate images
├── best_new.pt             # License plate detection model
├── tracker.py              # Object tracking module
├── Violation.py            # main implementation
├── requirements.txt
└── README.md


### ⚙️ Working Principle

1. Vehicle Detection

Vehicles are detected in each video frame using a pre-trained YOLOv8 model.

2. Vehicle Tracking

Detected vehicles are assigned unique IDs and tracked across frames.

3. Traffic Signal Monitoring

The system maintains the current signal state (RED/GREEN).

4. Violation Detection

When a vehicle crosses the predefined stop line during a RED signal, it is marked as a violator.

5. License Plate Recognition

The violating vehicle's license plate is detected and extracted using a custom YOLO model.

6. OCR Processing

EasyOCR reads the license plate number from the extracted plate image.

7. Evidence Storage

The system automatically saves:

* Vehicle violation image
* License plate image
* Recognized license plate number

### 🚀 Installation

Clone the Repository

git clone https://github.com/your-username/Vehicle-Rules-Violation-Detection.git
cd Vehicle-Rules-Violation-Detection

Install Dependencies

pip install -r requirements.txt

Required Packages

pip install ultralytics opencv-python easyocr pandas numpy

### ▶️ Usage

1. Place the input video in the project directory.
2. Update the video path in the script.
3. Run the application:

python violation.py

4. Detected violations and license plates will be saved automatically.

### 📸 Sample Output

Violation Detection

Violation Detected!
Vehicle ID: 12
Signal: RED

License Plate Recognition

Detected Plate: TN01AB1234


### 📈 Future Enhancements

* Speed violation detection
* Vehicle database integration
* Automatic e-challan generation

### 🎯 Applications

* Smart Traffic Management
* Automated Traffic Enforcement
* Intelligent Transportation Systems (ITS)
* Smart City Surveillance
* Traffic Monitoring and Analytics

👨‍💻 Author

**Aruna Arunachalam**


