import os
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import networkx as nx
import math
from ultralytics import YOLO  # Ensure you have the latest YOLO version installed
import threading
import queue
from collections import Counter

# Load YOLO model
yolo_model = YOLO("yolo11x.pt")  # Replace with your YOLO model path

# Set minimum confidence threshold
CONFIDENCE_THRESHOLD = 0.3  # Lowered for better detection

# Define class names for YOLO
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "dog",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "animal"
]  # Add more class names if needed

# Function to detect objects in a video frame
def detect_objects(frame):
    # Resize frame for faster processing while keeping aspect ratio
    h, w, _ = frame.shape
    scale_factor = 640 / max(h, w)
    resized_frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))

    # Run YOLO detection
    results = yolo_model(resized_frame)
    detections = []
    if hasattr(results[0], 'boxes'):
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()

            # Filter detections based on confidence threshold
            if conf >= CONFIDENCE_THRESHOLD and int(cls) < len(class_names):
                x1, y1, x2, y2 = int(x1 / scale_factor), int(y1 / scale_factor), int(x2 / scale_factor), int(y2 / scale_factor)
                detections.append((x1, y1, x2, y2, int(cls)))
    return detections

# Count objects detected in the frame
def count_objects(detections):
    counts = Counter(cls for (_, _, _, _, cls) in detections)
    summary = ", ".join(f"{counts[cls]} {class_names[cls]}" for cls in counts if cls < len(class_names))
    return summary

# Create a dynamic graph based on detected objects
def create_graph_from_objects(detections):
    G = nx.Graph()
    centers = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2, cls) in detections]

    # Add nodes with class names
    for i, (x1, y1, x2, y2, cls) in enumerate(detections):
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        G.add_node(i, label=class_name, cls=cls)

    # Add edges with proximity-based relationships
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distance = math.sqrt((centers[i][0] - centers[j][0]) ** 2 +
                                 (centers[i][1] - centers[j][1]) ** 2)
            if distance < 200:  # Example proximity threshold
                relationship = "Near" if distance < 100 else "Far Apart"
                G.add_edge(i, j, relationship=relationship)
    return G

# Visualize the graph with object counts and edges
def visualize_graph(nx_graph, ax, object_summary):
    ax.clear()
    labels = nx.get_node_attributes(nx_graph, 'label')
    colors = ['red' if nx_graph.nodes[node]['cls'] == 0 else 'blue' for node in nx_graph.nodes]
    pos = nx.spring_layout(nx_graph, seed=42)  # Use fixed seed for consistent layout

    # Draw nodes and edges
    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        labels=labels,
        node_color=colors,
        node_size=2000,
        font_size=9,
        font_color="white",
        ax=ax
    )

    # Draw edges with labels if edges exist
    if nx_graph.edges:
        edge_labels = nx.get_edge_attributes(nx_graph, 'relationship')
        if edge_labels:
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_title(f"Graph Visualization\nDetected Objects: {object_summary}", fontsize=12)
    plt.pause(0.001)

# Multithreaded video processing
def process_video(video_path, queue):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip every other frame for faster processing
        if frame_count % 2 != 0:
            continue

        # Detect objects in the frame
        detections = detect_objects(frame)

        # Count objects for summary
        object_summary = count_objects(detections)

        # Draw bounding boxes on the frame with proper class names
        for (x1, y1, x2, y2, cls) in detections:
            class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Bounding box in blue
            cv2.putText(
                frame,
                class_name,  # Display class name
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),  # Font color: green
                2
            )

        # Pass frame, detections, and object summary to the queue for graph processing
        queue.put((frame, detections, object_summary))

    cap.release()
    queue.put(None)  # Signal the end of processing

# Main script
if __name__ == "__main__":
    video_path = "videos/cctvfootage.mp4"  # Replace with your video path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Queue for communication between threads
    frame_queue = queue.Queue()

    # Start video processing in a separate thread
    video_thread = threading.Thread(target=process_video, args=(video_path, frame_queue))
    video_thread.start()

    # Matplotlib setup for graph visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    cv2.namedWindow("Video Frame with Detected Objects", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Frame with Detected Objects", 640, 480)

    while True:
        item = frame_queue.get()
        if item is None:
            break

        frame, detections, object_summary = item

        # Create and visualize the graph
        nx_graph = create_graph_from_objects(detections)
        visualize_graph(nx_graph, ax, object_summary)

        # Display the current frame with bounding boxes
        cv2.imshow("Video Frame with Detected Objects", frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    plt.close(fig)
