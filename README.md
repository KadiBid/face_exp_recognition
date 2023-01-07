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
  * '*export_models.ipynb*' Jupyter notebook to export the winning models to Tensorflow.js.

* *app* contains the fronend web application the user will interact with.
  * '*index.html*' Main code of the frontend web application.
  * '*model/CNN*' Folder containing the TensorFlow.js version of the CNN model.

The dataset is NOT in the repository, but it is expected to be in '*AI/dataset*'.


## Development ##

Overall, the development of this project involved two main areas of focus: the machine learning model and the application that consumes it. 

The machine learning model was designed to identify the facial expression of the faces on given images, being able to provide accurate predictions on new, unseen data.

The application that consumes the machine learning model was developed using HTML and JavaScript, and was responsible for displaying the labeled faces in real-time as the video stream was captured. 

The following subsections describes both of them more in detail.


### Machine Learning ###

The development of a machine learning model requires the following steps:

1. Collecting a dataset of facial images with labeled expressions. This dataset will be used to train the recognition model.
2. Designing and training different machine learning models.
3. Testing and analysing the models on a separate dataset to evaluate their performance.

#### Dataset ####

It is often not practical to create your own dataset, especially when it comes to large and complex ones such as images of faces with labeled expressions. In these cases, it is common to use existing datasets that have been compiled and curated by other researchers or organizations.

In my case I finally decided to use the [fer2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). This dataset consists of 32,298 grayscale images of faces sized 48x48, with each image labeled with one of seven expressions: angry, disgust, fear, happy, sad, surprise, and neutral. I considered to use other datasets, such as the [Google FEC dataset](https://www.kaggle.com/datasets/amar09/facial-expression-comparison-fec-google), but I discard them due to composition complexity.

The fer2013 contains images like the ones showed below:



The dataset was splitted into two groups, the first one containing images for training and the other one containing images for testing. I decided to mix them all and use train_test_split() to make this disctinction.


#### Models ####

In order to pick he best possible model for my project I decided to evaluate four different models, three pretrained models using transfer learning and a model designed from scratch applying some learnings I got from Core Code School and some great Youtube divulgators. The models I used are the following:

Pretrained:
* *MobileNetV2* is a lightweight machine learning model for image classification and object detection that is designed to run efficiently on mobile devices. It uses a neural network architecture that is optimized for speed and accuracy, making it a popular choice for mobile and edge applications.
* *ResNet50* is a deep convolutional neural network model trained on the ImageNet dataset. It is designed to recognize and classify objects in images and has achieved state-of-the-art performance on the ImageNet classification task. It is widely used in a variety of image recognition and computer vision applications.
* *VGG16* is a convolutional neural network model trained on the ImageNet dataset. It is characterized by its deep architecture, with 16 layers of convolutional and fully-connected layers, and is known for its good performance on image classification tasks. It is often used as a baseline model for comparison with other image recognition models.

From scratch:
* Convolutional model based on learnings from the following links:
  - [Redes Neuronales Convolucionales - Clasificación avanzada de imágenes con IA / ML (CNN)](https://www.youtube.com/watch?v=4sWhhQwHqug) by Ringa Tech.
  - [¡Redes Neuronales CONVOLUCIONALES! ¿Cómo funcionan?](https://www.youtube.com/watch?v=V8j1oENVz00) by DotCSV

To define and train the models I created a toolset based on Jupyter Notebook; '*generate_models.ipynb*'.





### Application ###


## References ##

* https://www.youtube.com/watch?v=4sWhhQwHqug (Convolucional model)

* https://www.youtube.com/watch?v=DbwKbsCWPSg (Webcam + Tensorflow.js)

* https://medium.com/@joomiguelcunha/lets-play-with-chrome-s-face-detection-api-ca13017a958f (FaceDetection on Chrome -experimental-)
