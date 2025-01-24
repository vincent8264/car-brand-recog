import os
import sys
import argparse
from PIL import Image
from functions.detector import detect_and_crop_cars
from functions.classifier import predict_car_brand
from functions.drawer import draw_box_and_label

def process_image(input_path, output_path):
    # Skip directories and non-image files
    if not os.path.isfile(input_path): 
        return
    if not input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Skipping non-image file: {input_path}")
        return

    try:
        # Load the image
        input_image = Image.open(input_path).convert("RGB")
        
        # Detect and crop the cars using detector
        cars_bbox, cars_image = detect_and_crop_cars(input_image)
        
        # Predict the brands using classification models
        predicted_brands, predicted_probs = predict_car_brand(cars_image)
        
        # Draw and save the predicted results
        input_image = draw_box_and_label(input_image, cars_bbox, predicted_brands, predicted_probs)
        input_image = Image.fromarray(input_image)
        input_image.save(output_path)
        print(f"Processed image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    # Input parser
    parser = argparse.ArgumentParser(description="Detect car brands in an image.")
    parser.add_argument(
        "--input",
        default=".",
        help="Path to the input image or folder (default is current folder)"
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Path to the output folder (default will create a folder './output')."
    )
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model.pt files are downloaded
    if not (os.path.isfile('functions/models/model_ft.pt') 
            or os.path.isfile('functions/models/model_tl.pt')):
        print("Model files not found. Please download the .pt model files and make sure they're in the /models folder")
        sys.exit()
        
    if os.path.isdir(input_path):
        print(f"Processing all images in folder: {input_path}")
        input_dir = input_path
        for file_name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, file_name)
            output_file_name = 'detected_' + file_name
            output_path = os.path.join(output_dir, output_file_name)
            process_image(input_path, output_path)        
            
    elif os.path.isfile(input_path):
        print(f"Processing single image: {input_path}")
        file_name = os.path.basename(input_path)
        output_file_name = 'detected_' + file_name
        output_path = os.path.join(output_dir, output_file_name)
        process_image(input_path, output_path)
        
    else:
        print(f"Invalid input path: {input_path}")