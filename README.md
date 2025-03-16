# moving-object-detection
Moving Object Detection from Video Using Hybrid CNN-GNN Model

Overview
This project focuses on moving object detection in CCTV surveillance footage using a hybrid CNN-GNN model. Traditional methods like YOLO and CNNs provide strong feature extraction but struggle with understanding object relationships. By integrating Graph Neural Networks (GNNs), our approach enhances detection robustness and spatial awareness.

Key Features
Hybrid CNN-GNN Model: CNN extracts object features, while GNN models spatial relationships.
YOLO-based Detection:
YOLO11x.pt – Used for high-accuracy object detection.
YOLO8x.pt – Used for faster but less detailed detection.
CCTV Surveillance Focus: Designed for real-world security applications.
YOLO Integration: Used for initial object detection before refining results with GNN.
Deep Learning-based Detection: Trained on real-world video datasets for improved generalization.

System Workflow
Preprocessing: Frames extracted from CCTV footage.
YOLO Detection:
YOLO11x.pt for accurate detection in complex environments.
YOLO8x.pt for faster inference when accuracy is not the primary concern.
CNN Feature Extraction: Extracts deep visual features from detected objects.
GNN Relationship Mapping: Establishes object relationships for better detection.
Final Object Classification: Output refined bounding boxes with detected objects.

Installation & Setup
Requirements
Python 3.x
OpenCV
TensorFlow / PyTorch
DeepFace
NetworkX (for GNN processing)

Installation

git clone https://github.com/yourusername/moving-object-detection.git  
cd moving-object-detection  
pip install -r requirements.txt  

Usage
To run the model on sample CCTV footage:

Run cnn_gnn_video.py for object detection in cctv footage 
Run new_cnn_gnn.py for object detection in .jpg files

Future Enhancements
Optimization for real-time edge deployment.
Behavior-based object recognition.
Improved handling of occlusions in crowded scenes.

Contributors
Khushi Patel (My Github: Khushi2099)




