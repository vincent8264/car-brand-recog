# Car Brand Recognizer
Detects the brands of cars in an input image using an object detection model and a classification model. Supports both single-image and batch processing.

(title image)


### Dependencies
torch  
torchvision  
PIL  
opencv  
numpy  

### Usage
#### Single image
```bash
python main.py --input path/to/input_image.jpg --output path/to/output_image.jpg
```

#### Multiple images in a folder
```bash
python main.py --input path/to/input_folder --output path/to/output_folder
```

### Demo

## Details
### Basic struture

The system uses three models in two steps. The first step uses a Retinanet model with pre-trained parameters that detects cars in the input image. The second step predicts the brands of the cars detected in the image, using ensembling of the two other classification models, a fine-tuned CNN and a transfer-learned efficientnet model. 

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

The classification models both have about 80% accuracy (f1 score) on the validation set. 

### Dataset
Images used to train the models are from three datasets:  
1. Stanford cars dataset  
https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

3. The Car Connection Picture Dataset  
https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper

4. Copyright-free images downloaded myself from the following websites  
[pexels.com  ](https://www.pexels.com/)  
[pixabay.com](https://www.pixabay.com/)  
[unsplash.com](https://unsplash.com/)  

