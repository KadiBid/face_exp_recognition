# Facial Expression Recognition

Author: Khadija El Abidi <kadi.elabidi@gmail.com>

| ℹ️ This is my Final Project for the '[Big Data y Machine Learning](https://www.corecode.school/bootcamp/bdml)' bootcamp on the [Core Code School](https://www.corecode.school/) |
| --- |

## About My Project

The objective of this final project is to use the skills and knowledge gained during the Big Data y Machine Learning bootcamp to create a complete and functional facial expression recognition system. This project serves as a capstone to the bootcamp, allowing me to demonstrate my understanding of the various concepts and techniques covered throughout the program.

Facial expression recognition is an interesting topic in the field of computer vision and machine learning.  It involves building a system that can analyze the emotions of a person from their facial expressions. This can be useful in a variety of applications, such as detecting emotional responses to marketing campaigns, improving customer service interactions, or even detecting emotional states in people with disabilities who have difficulty communicating their feelings, etc.

## Product ##

The product of this final project is a web application that utilizes the video stream from a user's webcam to detect and label faces in the video with their corresponding emotions. The app is designed to be easy to use, by just launching the browser.


<p align="center">
  <img src = "https://user-images.githubusercontent.com/112415704/210655036-fbdd37c1-ce72-4f08-8800-149504f42ea6.png">
</p>

## Architecture ##

The architecture is a front-end web application (HTML+JS) with everything self-contained.

1. Webcam (using javascript)
2. FaceDetector (using an experimental Chrome feature)
3. Machine Learning model (converted to javascript using [TensorFlow.js](https://www.tensorflow.org/js))

<p align="center">
  <img src = "https://user-images.githubusercontent.com/112415704/210812370-5266b479-c21d-4f07-9a67-e41cf57e63f2.png">
</p>

| ℹ️ This is my final architecture, but during the project I had a lot of doubts and made many changes as I learned new things.  |
| --- |


## About The Repository ##

That repository contains two folders:

* *AI* contains all my toolset (jupyter notebooks) to generate and analyze the machine learning models.
  * '*generate_models.ipynb*' Jupyter notebook to define and train different models (CNN, MobileNet, ResNet50 and VGG16).
  * '*analyze_models.ipynb*' Jupyter notebook to show the performance of the winning model (CNN).
  * '*export_models.ipynb*' Jupyter notebook to export the winning model to Tensorflow.js.

* *app* contains the fronend web application the user will interact with.
  * '*index.html*' Main code of the frontend web application.
  * '*model/CNN*' Folder containing the TensorFlow.js version of the CNN model.

The dataset is not in the repository, but it is expected to be in '*AI/dataset*'.


## Development ##





### About Dataset: 2

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

###### Dataset link:  https://www.kaggle.com/datasets/msambare/fer2013


### Steps: 


    1. Install dataset from kaggle: https://www.kaggle.com/datasets/msambare/fer2013
    2. Convert folders with image and labels to a dataframe train or test with pixels and labels
    3. Train data with model tensorflow Keras
    4.


