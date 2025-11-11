import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import requests
import os

def get_imagenet_labels():
    """Get ImageNet class labels"""
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    try:
        response = requests.get(url, timeout=10)
        labels = response.json()
        return {int(key): value[1] for key, value in labels.items()}
    except:
        return {0: "background"}

def get_available_models():
    """Return list of available models"""
    return ["ResNet50", "VGG16", "AlexNet", "DenseNet121", "Custom Model"]

def load_model(model_name="ResNet50"):
    """Load pre-trained model and preprocessing function"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        if model_name in ["ResNet50", "Custom Model"]:
            model = models.resnet50(pretrained=True)
        elif model_name == "VGG16":
            model = models.vgg16(pretrained=True)
        elif model_name == "AlexNet":
            model = models.alexnet(pretrained=True)
        elif model_name == "DenseNet121":
            model = models.densenet121(pretrained=True)
        else:
            model = models.resnet50(pretrained=True)
        
        model.eval()
        return model, preprocess
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        model = models.resnet50(pretrained=True)
        model.eval()
        return model, preprocess

def predict(model, image, top_k=5):
    """Make prediction and return top classes"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    imagenet_labels = get_imagenet_labels()
    top_classes = [imagenet_labels.get(idx, f"class_{idx}") for idx in top_indices.numpy()]
    
    return output, top_classes, top_probs.numpy()