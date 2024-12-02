import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolo11n.pt')

# Load and preprocess the image
filename = "apple.png"
img = cv2.imread(filename)
if img is None:
    raise FileNotFoundError(f"Image file '{filename}' not found.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run YOLO object detection
results = model(img_rgb)

# Load the MiDaS model for depth estimation
model_type = "DPT_Large"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas = torch.hub.load("intel-isl/MiDaS", model_type, device=device)
midas.eval()

# Transform the image for depth estimation
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
input_batch = transform(img_rgb).to(device)

# Predict depth
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Normalize the depth map for display
output = prediction.cpu().numpy()
normalized_output = (output - np.min(output)) / (np.max(output) - np.min(output))
depth_map = (normalized_output * 255).astype(np.uint8)
depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)  # Apply color map

# Draw YOLO detections on the depth map
for box in results[0].boxes.data:  # Iterate through each detected object
    x1, y1, x2, y2, conf, cls = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
    
    # Calculate center of the rectangle
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Extract depth at the center of the bounding box
    depth_value = normalized_output[center_y, center_x] * 255  # Convert to 8-bit scale
    depth_text = f"Depth: {depth_value:.2f}"
    
    # Draw rectangle and label
    cv2.rectangle(depth_map_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
    cv2.putText(depth_map_colored, depth_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display images
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(normalized_output, cmap='inferno')
axes[1].set_title("Depth Map")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2RGB))
axes[2].set_title("Depth Map with YOLO Detections")
axes[2].axis("off")

plt.tight_layout()
plt.show()
