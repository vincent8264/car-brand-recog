# Car Brand Recognizer
A Python-based tool that detects cars and identifies their brands in images using deep-learning object detection and classification models.  

![title image](/samples/example1.jpg)

The models are trained to predict 16 common car brands:  
Audi, BMW, Chevrolet, Ford, Honda, Hyundai, Jeep, Kia, Lexus, Mercedes-Benz, MINI, Nissan, Subaru, Tesla, Toyota, Volkswagen  

## Table of Contents
- [Car Brand Recognizer](#car-brand-recognizer)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Single Image](#single-image)
  - [Multiple Images](#multiple-images-in-a-folder)
- [Demo](#demo)
- [Details](#details)
  - [Basic Structure](#basic-structure)
  - [Dataset](#dataset)

### Dependencies
pytorch 2.5+ 
torchvision  
PIL  
opencv  
numpy  

CUDA is optional, the code will automatically run to model on cpu if cuda isn't detected

### Usage
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

### Demo
Example results showing bounding boxes and detected car brands.  

![example image](/samples/example2.jpg)
![example image](/samples/example3.jpg)
![example image](/samples/example8.jpg)
![example image](/samples/example9.jpg)
![example image](/samples/example10.jpeg)
![example image](/samples/example11.png)

More examples can be found in the samples folder

## Details
### Basic struture

The system uses three models in two steps. The first step uses a Retinanet model with pre-trained COCO dataset parameters that detects cars in the input image. The second step predicts the brands of the cars detected in the image, using ensembling of the two other classification models, a fine-tuned CNN and a transfer-learned efficientnet model. 

The classification models both have about 80% accuracy (f1 score) on the validation set. 



### Dataset
Car images used to train the models are from three datasets, with car models mostly between 2005~2020.
1. Stanford cars dataset  
https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

2. The Car Connection Picture Dataset  
https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper

3. Copyright-free images downloaded myself  
[pexels.com  ](https://www.pexels.com/)  
[pixabay.com](https://www.pixabay.com/)  
[unsplash.com](https://unsplash.com/)  

