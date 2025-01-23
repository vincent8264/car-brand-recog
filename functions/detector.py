import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.transforms import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CAR_CLASS = 3 
TRUCK_CLASS = 8
DETECTION_THRESHOLD = 0.5 # Minimum output score from the object detection model
SIZE_THRESHOLD = 0.005  # Minimum relative size of detected car to image
BORDER_THRESHOLD = 0.1  # Minimum relative distance to the border
BLOCKED_THRESHOLD = 0.4

def is_blocked(box, other_boxes):
    x_min, y_min, x_max, y_max = map(int, box)
    box_area = (x_max - x_min) * (y_max - y_min)
    for other_box in other_boxes:
        ox_min, oy_min, ox_max, oy_max = map(int, other_box)
        
        # Calculate intersection area
        inter_x_min = max(x_min, ox_min)
        inter_y_min = max(y_min, oy_min)
        inter_x_max = min(x_max, ox_max)
        inter_y_max = min(y_max, oy_max)
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # If a fraction of box area is in other boxes, then it's blocked
        if inter_area / box_area > BLOCKED_THRESHOLD:  
            return True
        
    return False

def detect_and_crop_cars(input_image):
    cars_image, cars_coords = [], []
    
    # Load the object detection model
    model = retinanet_resnet50_fpn_v2(weights='COCO_V1')
    model.to(device)
    model.eval()

    # Inference
    image_tensor = F.to_tensor(input_image).unsqueeze(0)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)
        
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    image_width, image_height = input_image.size
    
    car_boxes = [
        boxes[i].tolist()
        for i in range(len(labels))
        if (labels[i] == CAR_CLASS or labels[i] == TRUCK_CLASS) and scores[i] > DETECTION_THRESHOLD
    ]
    
    # Filter out unsuited cars
    filtered_boxes = []
    for box in car_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Check size threshold
        if (box_width * box_height) / (image_width * image_height) < SIZE_THRESHOLD:
            continue

        # Check border threshold
        if (x_mid / image_width < BORDER_THRESHOLD or 
            y_mid / image_height < BORDER_THRESHOLD or
            x_mid / image_width > 1 - BORDER_THRESHOLD or 
            y_mid / image_height > 1 - BORDER_THRESHOLD):
            continue

        # Check if blocked
        if is_blocked(box, filtered_boxes):
            continue

        filtered_boxes.append(box)

    # Crop the detected cars
    for box in filtered_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        car = input_image.crop((x_min, y_min, x_max, y_max))
        cars_image.append(car)
        cars_coords.append([x_min, y_min, x_max, y_max])
        
    return cars_coords, cars_image