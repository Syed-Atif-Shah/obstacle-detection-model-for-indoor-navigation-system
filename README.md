# obstacle-detection-model-for-indoor-navigation-system
Obstacle Detection with YOLOv8 and Flask
A web application that detects obstacles in images using a custom-trained YOLOv8 model, deployed via a Flask backend.
Key Features
✔ Custom Model: Fine-tuned YOLOv8 on annotated obstacle dataset
✔ Web Interface: Upload images and view detections with bounding boxes
✔ Optimized: Model exported to ONNX for faster inference
Tech Stack
Backend: Python, Flask
Computer Vision: Ultralytics YOLOv8, OpenCV
Deployment: Localhost (Flask)
Usage
Clone the repo
Install dependencies: pip install -r requirements.txt
Run: python app.py
Dataset: Roboflow | Model: best.onnx

