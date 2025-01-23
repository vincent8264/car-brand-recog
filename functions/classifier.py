from torchvision.transforms.v2 import ToImage,ToDtype,Resize,Compose,Normalize
from functions.model_list import CNN, CNNTransfer
import torch
import json
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.deterministic = True

# Open car brand mapping file
with open("functions/class_id.json", 'r') as file:
    mapping = json.load(file)
    id_to_class = mapping['id_to_class']

def get_predictions(model, image, transforms):
    img_unsqueezed = transforms(image).unsqueeze(0)
    img_unsqueezed = img_unsqueezed.to(device)
    pred = model(img_unsqueezed)
    probs = torch.exp(pred)
    return probs.cpu()

def ensemble(probs_1, probs_2, method='1'):
    if method == 'product':
        product_probs = probs_1 * probs_2
        product_probs = product_probs.sqrt()
        return product_probs / product_probs.sum()
    elif method == 'arithmatic':
        return (probs_1 + probs_2) / 2
    elif method == 'max':
        product_probs = probs_1 * probs_2
        product_probs = product_probs / product_probs.sum()
        stacked = torch.stack([probs_1, probs_2, product_probs], dim=0)
        max_probs, _ = torch.max(stacked, dim=0)
        return max_probs
    elif method == '1':
        return probs_1
    elif method == '2':
        return probs_2

def predict_car_brand(cars):
    # Load models
    model1_name = 'functions/models/model_ft.pt'
    model2_name = 'functions/models/model_tl.pt'
    model1 = CNN(classes = len(id_to_class))
    model2 = CNNTransfer(classes = len(id_to_class))
    
    # Initialize model 1
    transforms1 = Compose([
        Resize([96,128]),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(           
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])
    model1.load_state_dict(torch.load(model1_name,weights_only=True))
    model1.to(device)
    model1.eval()
    
    # Initialize model 2 
    transforms2 = Compose([
        Resize([384,384]),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(           
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])    
    model2.load_state_dict(torch.load(model2_name,weights_only=True))
    model2.to(device)
    model2.eval()
    
    # Predict all the images from input
    predicted_brands, predicted_probs = [], []
    for img in cars:
        with torch.no_grad():
            probs_1 = get_predictions(model1, img, transforms1)        
            probs_2 = get_predictions(model2, img, transforms2)        
        
        # Ensemble probabilities
        probs = ensemble(probs_1, probs_2, method = 'max')
        
        # Get the highest label
        label = torch.argmax(probs,dim=1).item()
        predicted_prob = probs[0, label].item()
        predicted_brand = id_to_class[str(label)]  
        
        predicted_brands.append(predicted_brand)
        predicted_probs.append(predicted_prob)
        
    return predicted_brands, predicted_probs