import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import logging
import requests
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Inference will run on device: {device}")

# Model architecture
def net(num_classes=5):
    logger.info("Defining model")
    model = models.resnet50(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features,num_classes))
    return model


# Load the model
def model_fn(model_dir):    
    # Load the saved state_dict
    logger.info(f"Loading model from: {model_dir}")
    model = net(num_classes=5)

    model_path = os.path.join(model_dir, "model.pth")
    
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
        
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    return model


def input_fn(request_body, content_type):
    logger.debug(f'Content type of request is: {content_type}')
    logger.debug(f'Request body type is: {type(request_body)}')
    
    if content_type in ['image/jpeg', 'image/png', 'application/x-image']:
        image = Image.open(io.BytesIO(request_body)).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    logger.info("Running inference")
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
        probs = torch.nn.functional.softmax(output, dim=1)
    return probs.cpu()

# Postprocess the output
def output_fn(prediction, accept):
    
    probs = prediction.squeeze().numpy()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    
    idx_to_class = {0: "Northern Watersnake", 1: "Common Garter snake", 2: "DeKay's Brown snake", 3: "Black Rat snake", 4: "Western Diamondback rattlesnake"}  
    response = {
        "class_index": prediction,
        "class_label": idx_to_class.get(prediction, "Unknown")
    }

    result = {
        "class_index": top_idx,
        "class_label": idx_to_class.get(top_idx, "Unknown"),
        "confidence": round(top_prob, 3)
    }


    if accept == "application/json":
        return json.dumps(result), accept
    else:
        return str(result), "text/plain"