# Car Brand Recognizer

![title image](/samples/example1.jpg)

## Table of Contents
- [Car Brand Recognizer](#car-brand-recognizer)
- [Introduction](#introduction)
- [Usage](#usage)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Running the program](#running-the-program)
    - [Single Image](#single-image)
    - [Multiple Images](#multiple-images-in-a-folder)
  - [Output](#output)
- [Examples](#examples)
- [Details](#details)
  - [Basic Structure](#basic-structure)
  - [detector.py](#detectorpy)
  - [classifier.py](#classifierpy)
  - [drawer.py](#drawerpy)
- [Dataset](#dataset)

## Introduction
This is a Python-based tool for detecting cars and identifying their brands in images, utilizing deep-learning object detection and classification models

The models are trained to predict 16 common car brands:  
Audi, BMW, Chevrolet, Ford, Honda, Hyundai, Jeep, Kia, Lexus, Mercedes-Benz, MINI, Nissan, Subaru, Tesla, Toyota, Volkswagen  

I made this as a practice in deep learning, computer vision, and Python development.

## Usage
### Dependencies
pytorch 2.5+  
torchvision 0.20  
PIL 11.0  
opencv-python 4.11  
numpy 2.2.2 

CUDA is optional, the code will automatically run the model on cpu if cuda isn't available

### Installation

```bash
git clone https://github.com/vincent8264/car-brand-recog.git
```
Due to GitHub's file size limits, the models must be downloaded separately.  

You can download the models on [google drive](https://drive.google.com/drive/folders/1KsxsLipO8j8h9q5YTSVhAheix0zfj4UJ), and place the .pt files in the ./functions/models folder like this:  

│── car_recog.py  
│── functions  
│  │── models  
│  │  │── PUT_THE_MODELS_HERE.txt  
│  │  │── model_ft.pt  
│  │  │── model_tl.pt  


### Running the program
Inputs with .jpg .jpeg .png are supported, other file formats will be ignored. 

#### Single image
```bash
python car_recog.py --input path/to/input_image.jpg --output path/to/output_folder
```
#### Multiple images in a folder
```bash
python car_recog.py --input path/to/input_folder --output path/to/output_folder
```
  
If `--input` is omitted, the program will scan the current directory for supported image files.  
If `--output` is omitted, results will be saved in a new `./output` folder in the current directory.

### Output 
After running the program, the output folder will contain annotated images with bounding boxes and labels, adding "detected" to the image name:

│── output folder  
│  │── detected_input1.jpg  
│  │── detected_input2.jpeg  
│  │── detected_input3.png  

## Examples

![example image](/samples/example2.jpg)
![example image](/samples/example3.jpg)
![example image](/samples/example8.jpg)
![example image](/samples/example9.jpg)
![example image](/samples/example10.jpeg)
![example image](/samples/example11.png)

More examples can be found in the samples folder along with the sources of the images

## Details
### Basic structure

The system uses three models in two steps. The first step uses a Retinanet model with pre-trained COCO dataset parameters that detects cars in the input image. The second step predicts the brands of the cars detected in the image, using ensembling of the two other classification models, a fine-tuned CNN and a transfer-learned efficientnet model. 

The following description explains each step in detail.  

### detector.py

Uses retinanet_resnet50_fpn_v2 to detect cars in the input image, for its balance between accuracy and computational needs. All objects that are detected as labels "car" and "truck" are considered as cars. This is because pick-ups, which most people would call them "cars", are in the "truck" class in the COCO dataset. Unfortunately, this causes other actual "trucks" like semis to be included.

The cars in the images then go through additional checks to filter out those that are: 
- Too close to the border
- Too small
- Blocked by other cars

All the detected cars that satisfy these thresholds are then cropped, and send to the classifier.

### classifier.py

The classification step uses two models to predict the brand of the cropped cars. The first model is a fine-tuned CNN which uses a VGG-like architecture with fewer units and layers. The second model is a transfer-learned model, with efficientnet_v2_s as the base model. The classification models both have about 80% accuracy (f1 score) on the validation set. 

After prediction, the output probabilities of each class are ensembled. The default ensembling method takes the maximum probability among the following three:
1. probabilities of each class from model 1
2. probabilities of each class from model 2
3. element-wise product of 1. and 2.

This is similar to using the class with the highest probability, but if both models also think another class has a significant amount of chances, the final prediction will be that class instead.

### drawer.py

Finally, in this part, the bounding boxes of cars are rendered, along with the predicted class and ensembled probabilities as labels. The default color for the boxes and labels are green for more visibility.


## Dataset
Car images used to train the classification models are from three datasets, with car models mostly between 2005~2020.  

1. Stanford cars dataset  
https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

2. The Car Connection Picture Dataset  
https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper
https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset/data

3. Copyright-free images downloaded myself  
[pexels.com  ](https://www.pexels.com/)  
[pixabay.com](https://www.pixabay.com/)  
[unsplash.com](https://unsplash.com/)  



