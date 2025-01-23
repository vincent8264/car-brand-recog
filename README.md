# Car Brand Recognizer
Detects the brands of cars in an input image using an object detection model and a classification model. Supports both single-image and batch processing.  

![title image](/samples/example1.jpg)


The models are trained to predict 16 common car brands below:

Audi  
BMW  
Chevrolet  
Ford  
Honda  
Hyundai  
Jeep  
Kia  
Lexus  
Mercedes-Benz  
MINI  
Nissan  
Subaru  
Tesla  
Toyota  
Volkswagen  

### Dependencies
pytorch 2.5+ 
torchvision  
PIL  
opencv  
numpy  

### Usage
#### Single image
```bash
cd diretory/to/car_recog
python car_recog.py --input path/to/input_image.jpg --output path/to/output_folder
```
#### Multiple images in a folder
```bash
cd diretory/to/car_recog
python car_recog.py --input path/to/input_folder --output path/to/output_folder
```

If --input and --output are not specified, it will default to the current directory, and will create a ./output folder

### Demo
![example image](/samples/example2.jpg)
![example image](/samples/example3.jpg)
![example image](/samples/example8.jpg)
![example image](/samples/example9.jpg)
![example image](/samples/example10.jpg)
![example image](/samples/example11.jpg)

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

