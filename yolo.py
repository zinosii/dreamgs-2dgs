from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["chair_table.jpeg"])  # return a list of Results objects

boxes = 0

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

    print(boxes.xyxy)


import cv2
import torch

image_path = "chair_table.jpeg"
image = cv2.imread(image_path)

boxes = (boxes.xyxy).cpu().numpy().astype(int)

for i, (x1, y1, x2, y2) in enumerate(boxes):
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(f"fig{i+1}.jpg", cropped_image)

print("Images have been saved.")

