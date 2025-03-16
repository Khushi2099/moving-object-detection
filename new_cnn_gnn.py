import os
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import networkx as nx
import pickle

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the AutoencoderCNN model (unchanged)
class AutoencoderCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU()
        )
        # Latent space
        self.latent = torch.nn.Linear(512 * 8 * 8, 16) 
        self.latent_to_decoder = torch.nn.Linear(16, 512 * 8 * 8)

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        z = self.latent(x)        # Latent representation
        x = self.latent_to_decoder(z)
        x = x.view(x.size(0), 512, 8, 8)  # Reshape for decoder
        x = self.decoder(x)
        return z


# Function to detect faces in the entire image without cropping
def detect_faces(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    frame = cv2.imread(image_path)
    
    if frame is None:
        raise ValueError(f"Error loading image: {image_path}. Ensure the file exists and is a valid image.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert only if frame is valid
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame, faces

# Create a dynamic graph based on detected faces
def create_graph_from_faces(faces):
    G = nx.Graph()
    for i in range(len(faces)):
        G.add_node(i, label=f"Face {i+1}")
    # Example: Add edges between all nodes (fully connected graph)
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            G.add_edge(i, j)
    return G

# Visualize the graph
def visualize_graph(nx_graph, predictions):
    # Assign predictions as node attributes
    for i, node in enumerate(nx_graph.nodes):
        nx_graph.nodes[node]['prediction'] = predictions[i].item()

    # Color nodes based on predictions
    colors = ['red' if nx_graph.nodes[node]['prediction'] == 0 else 'blue' for node in nx_graph.nodes]

    # Draw the graph
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(nx_graph)  # Generate layout
    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        node_color=colors,
        node_size=500,
        font_size=10,
        font_color="white",
        ax=ax
    )
    plt.title("Graph Visualization with Predictions")
    plt.show()
    plt.close(fig)

# Main script
if __name__ == "__main__":
    # Detect faces in the image
    image_path = "photos/group.jpg"
    detected_image, faces = detect_faces(image_path)

    # Display detected faces
    cv2.imshow("Detected Faces", detected_image)
    print(f"Number of faces detected: {len(faces)}")

    # Create a graph dynamically based on detected faces
    nx_graph = create_graph_from_faces(faces)

    # Mock predictions for the graph
    predictions = torch.randint(0, 2, (nx_graph.number_of_nodes(),))

    # Visualize the graph
    visualize_graph(nx_graph, predictions)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
