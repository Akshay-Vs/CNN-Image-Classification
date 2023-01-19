import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import urllib

# Load Model
model = models.load_model('image_classification.model')

# Load and preprocess image
url = input("Enter url to the image: ")

# Open the image from the URL
with urllib.request.urlopen(url) as url:
    # Read the image data
    img_array = np.array(bytearray(url.read()), dtype=np.uint8)

# Decode the image data using OpenCV
img = cv.imdecode(img_array, cv.IMREAD_COLOR)

# Check if the image was successfully loaded
if img is None:
    print("Error: Failed to load image")
else:
    # Convert image to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Resize the image
    img = cv.resize(img, (32, 32))
    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    print(f"Prediction is {class_names[index]}")
