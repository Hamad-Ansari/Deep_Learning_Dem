import cv2
import numpy as np
from PIL import Image

def enhance_image(image, enhancement_type):
    """Apply image enhancement"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if enhancement_type == "Sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)
    elif enhancement_type == "Contrast":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif enhancement_type == "Brightness":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return image

def preprocess_image(image, size=(224, 224)):
    """Basic image preprocessing"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.resize(size)