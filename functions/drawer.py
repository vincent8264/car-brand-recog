import cv2
import numpy as np

GREEN = (18, 127, 15)
GRAY = (218, 227, 218)
RED = (220, 20, 60)
line_scale_factor = 0.003
font_scale_factor = 0.0007

def draw_box(image, box):
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = box
    color = GREEN
    
    thickness = max(1, int(line_scale_factor * ((image_width + image_height) / 2)))
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=thickness)

    return image

def draw_label(image, box, text):
    image_height, image_width = image.shape[:2]
    x_min, y_min, x_max, y_max = box
    back_color = GREEN
    text_color = GRAY

    # Compute text size
    font_scale = font_scale_factor * ((image_width + image_height) / 2)
    thickness = max(1, int(font_scale * 3))
    font = cv2.FONT_HERSHEY_DUPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Text background
    back_tl = x_min, max(0, y_min - int(1.3 * text_h))
    back_br = x_min + text_w, y_min
    cv2.rectangle(image, back_tl, back_br, back_color, -1)

    # Show text
    text_tl = x_min, max(0, y_min - int(0.2 * text_h))
    cv2.putText(image, text, text_tl, font, font_scale,
                text_color, thickness=thickness, lineType=cv2.LINE_AA)    
    
    return image

def draw_box_and_label(image, cars_bbox, predicted_brands, predicted_probs):
    image = np.array(image)

    # Draw bounding box
    for i in reversed(range(len(cars_bbox))):
        image = draw_box(image, cars_bbox[i])

    # Draw label and probability
    for i in reversed(range(len(cars_bbox))):
        label = " ".join([predicted_brands[i], str(round(100 * predicted_probs[i],1))])
        image = draw_label(image, cars_bbox[i], label)
    
    return image
