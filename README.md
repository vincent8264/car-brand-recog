# Car Brand Recognizer
A Python-based tool that detects cars and identifies their brands in images using deep-learning object detection and classification models.  

![title image](/samples/example1.jpg)

The models are trained to predict 16 common car brands:  
Audi, BMW, Chevrolet, Ford, Honda, Hyundai, Jeep, Kia, Lexus, Mercedes-Benz, MINI, Nissan, Subaru, Tesla, Toyota, Volkswagen  

## Table of Contents
- [Car Brand Recognizer](#car-brand-recognizer)
- [Usage](#usage)
  - [Dependencies](#dependencies)
  - [Running the program](#running-the-program)
    - [Single Image](#single-image)
    - [Multiple Images](#multiple-images-in-a-folder)
- [Demo](#demo)
- [Details](#details)
  - [Basic Structure](#basic-structure)
  - [detector.py](#detector.py)
  - [classifier.py](#classifier.py)
  - [drawer.py](#drawer.py)
- [Dataset](#dataset)

## Usage
### Dependencies
pytorch 2.5+  
torchvision  
PIL  
opencv  
numpy  

CUDA is optional, the code will automatically run to model on cpu if cuda isn't detected

### Installation

```bash
git clone https://github.com/vincent8264/car-brand-recog.git
```
Due to the file size limits of github, a separate download of the models are required.  

You can download the models from [google drive](https://drive.google.com/drive/folders/1KsxsLipO8j8h9q5YTSVhAheix0zfj4UJ), and place the .pt files in the ./functions/models folder like this:  

│── car_recog.py  
│── functions  
│  │── models  
│  │  │── PUT_THE_MODELS_HERE.txt  
│  │  │── model_ft.pt  
│  │  │── model_tl.pt  


### Running the program
Inputs with .jpg .jpeg .png are supported, and the output images will include the predicted bounding boxes and labels.  

#### Single image
```bash
python car_recog.py --input path/to/input_image.jpg --output path/to/output_folder
```
#### Multiple images in a folder
```bash
python car_recog.py --input path/to/input_folder --output path/to/output_folder
```
  
If --input is omitted, the current directory is scanned for supported image files.  
If --output is omitted, the output images will be saved to a ./output folder.

## Demo
Example results showing bounding boxes and detected car brands.  

![example image](/samples/example2.jpg)
![example image](/samples/example3.jpg)
![example image](/samples/example8.jpg)
![example image](/samples/example9.jpg)
![example image](/samples/example10.jpeg)
![example image](/samples/example11.png)

More examples can be found in the samples folder!

## Details
### Basic struture

The system uses three models in two steps. The first step uses a Retinanet model with pre-trained COCO dataset parameters that detects cars in the input image. The second step predicts the brands of the cars detected in the image, using ensembling of the two other classification models, a fine-tuned CNN and a transfer-learned efficientnet model. 

The following description explains each steps in detail.

### detector.py

Uses retinanet_resnet50_fpn_v2 to detect cars in the input image, for its balance between accuracy and computational needs. All objects that are detected as labels "car" and "truck" are considered as cars. This is because pick-ups, which most people would call them "cars", are in the "truck" class in the COCO dataset. Unfortunately, this causes other actual "trucks" like semis to be included.

The cars in the images then pass through some checks to filter out cars that are: 
- Too close to the border
- Too small
- Blocked by other cars

All the detected cars that satisfy these thresholds are then cropped, and send to the classifier.

### classifier.py

The classification step uses two models to predict the brand of the cropped cars. The first model is a fine-tuned CNN model with a simplified VGG architecture. The second model is a transfer-learned model, with efficientnet_v2_s as the base model. The classification models both have about 80% accuracy (f1 score) on the validation set. 

After predicting, the output probabilities of each class are then ensembled. The default ensembling method takes the maximum probability among the following three:
1. probabilities of each class from model 1
2. probabilities of each class from model 2
3. element-wise product of 1. and 2.

This is similar to just use whichever class that has the highest probs, but if both models also think another class has a significant amount of chances, the final prediction will be that class instead.

### drawer.py

Finally, in this part, the bounding boxes of cars are rendered on the input image, with the predicted class and ensembled probabilities as the labels. 


## Dataset
Car images used to train the classification models are from three datasets, with car models mostly between 2005~2020.  

1. Stanford cars dataset  
https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

2. The Car Connection Picture Dataset  
https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper

3. Copyright-free images downloaded myself  
[pexels.com  ](https://www.pexels.com/)  
[pixabay.com](https://www.pixabay.com/)  
[unsplash.com](https://unsplash.com/)  

