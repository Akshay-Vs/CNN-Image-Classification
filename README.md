# CNN-Image-Classification
This project is an image classification model that uses the CIFAR-10 dataset and a convolutional neural network (CNN) to classify images into 10 different classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The trained model can then be used to predict the class of new images by inputting the image's URL.

## Requirements
- Python 3.x
- Tensorflow 2.x
- OpenCV
- Numpy
- Urllib

## Usage
- Run the train.py file to train the model and save it.
- Run the predict.py file to load the pretrained model and prompt the user to input the URL of an image. 
- The model will then classify the image and print the prediction.

## Model Architecture
- 2D convolutional layer with 32 filters and a kernel size of (3,3), using the ReLU activation function
- Max pooling layer with a size of (2,2)
- 2D convolutional layer with 64 filters and a kernel size of (3,3), using the ReLU activation function
- Max pooling layer with a size of (2,2)
- 2D convolutional layer with 64 filters and a kernel size of (3,3), using the ReLU activation function
- Flatten layer
- Dense layer with 64 neurons and the ReLU activation function
- Dense output layer with 10 neurons and the softmax activation function


The model is a sequential CNN model with 3 convolutional layers and 2 dense layers. Each convolutional layer is followed by a max pooling layer to reduce the spatial - dimension of the data. The input shape of the image is (32,32,3) and the output shape is (10,) representing the probability of the image belonging to each of the 10 classes. The model is trained using the ```Adam optimizer``` and the ```categorical cross-entropy loss``` function.

The dataset is loaded using the ```np.load()``` function and split into training and test sets using ```sklearn.model_selection.train_test_split()```. The image data is preprocessed by normalizing the pixel values between 0 and 1 using img = img / 255.0.

The input image is read using ```urllib.request.urlopen()``` and decoded using ```cv2.imdecode()```. The image is then resized to (32,32) and passed through the model for prediction. The class with the highest probability is chosen as the final prediction.
