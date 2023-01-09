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

<p align="center">
  <img src = "https://user-images.githubusercontent.com/112415704/211166777-06205b6b-bf3f-4319-bf5c-8469d0941235.jpg">
</p>

The dataset was splitted into two groups, the first one containing images for training and the other one containing images for testing. I decided to mix them all and use train_test_split() to make this disctinction.

| ℹ️ During the project I identfied some weaknesses in the dataset (images withou faces, images incorrectly labeled, etc) but they were afecting a small subset of images so I considered them negligible and didn't cleanup them. |
| --- |

#### Models ####

In order to pick he best possible model for my project I decided to evaluate four different models, three pretrained models using transfer learning and a model designed from scratch applying some learnings I got from Core Code School and some great Youtube divulgators. The models I used are the following:

From scratch:
* *Convolutional* model based on learnings from the following links:
  - [Redes Neuronales Convolucionales - Clasificación avanzada de imágenes con IA / ML (CNN)](https://www.youtube.com/watch?v=4sWhhQwHqug) by Ringa Tech.
  - [¡Redes Neuronales CONVOLUCIONALES! ¿Cómo funcionan?](https://www.youtube.com/watch?v=V8j1oENVz00) by DotCSV

Pretrained:
* *MobileNetV2* is a lightweight convolutional machine learning model for image classification and object detection that is designed to run efficiently on mobile devices. It uses a neural network architecture that is optimized for speed and accuracy, making it a popular choice for mobile and edge applications.
* *ResNet50* is a deep convolutional neural network model trained on the ImageNet dataset. It is designed to recognize and classify objects in images and has achieved state-of-the-art performance on the ImageNet classification task. It is widely used in a variety of image recognition and computer vision applications.
* *VGG16* is a convolutional neural network model trained on the ImageNet dataset. It is characterized by its deep architecture, with 16 layers of convolutional and fully-connected layers, and is known for its good performance on image classification tasks. It is often used as a baseline model for comparison with other image recognition models.

To define and train the models I created a tool based on Jupyter Notebook; '*generate_models.ipynb*'. After training all the models I got the following results:

**FROM SCRATCH**

**Convolucional Model:**

![photo_2023-01-08 23 34 10](https://user-images.githubusercontent.com/112415704/211374513-aaddc5bf-5306-4510-8e1c-f49a869a1167.jpeg)




**PRETRAINED **

**ResNet50:**

![photo_2023-01-08 21 16 32](https://user-images.githubusercontent.com/112415704/211374706-37f97716-e160-4545-9797-246731cc63cb.jpeg)


**MobileNetV2:**

<p align="center">
  <img src = "https://user-images.githubusercontent.com/112415704/211373999-13a20b75-c606-4161-a5f8-5d088e1f1afc.png">
</p>


**VGG16:**

<p align="center">
  <img src = "https://user-images.githubusercontent.com/112415704/211374441-bed943cb-41f1-468c-a20a-dc95967b6b31.png">
</p>

Pretrained models shows an slow training process, probably because I am labeling the pretrained data as non-trainable, so only the Dense layers I added at the end can be trained. 

I finally chose to use the convolutional model that I trained from scratch to be used on my product.

### Application ###

The aplication will be a web application, so I need to be able to use the model from Javascript. Fortunately there is a TensorFlow.js tool to convert the models, so I created a tool based on Jupyter Notebook to export the model to javascript: '*export_models.ipynb*'.

Based on the code presented in this tutorial:

- [Crea un clasificador de perros y gatos con IA, Python y Tensorflow - Proyecto completo](https://www.youtube.com/watch?v=DbwKbsCWPSg) by Ringa Tech.
 
I developed a basic application to capture the camera and pass the video images to the model and show the emotion detected, but the results were not very good. The reason this happened was because the positioning of the face in the webcam and in the dataset images was very different.


## Demo ##

https://user-images.githubusercontent.com/112415704/211371433-80f839f6-8bc5-4bc7-9fc2-c688cd240965.mp4



## Limitations ##

I was not able to incorporate into the product all 7 emotions. I limited it to 4; happy, surprise, neutral and sad. Most of them works pretty reliable for excepte of the sad emotion, wich is not working perfect for some people.



## Presenter View ##

https://user-images.githubusercontent.com/112415704/211383833-afe60388-8758-4a75-8dce-f92216218bba.mp4



## Renferenses ##

https://gist.github.com/ritwickdey/83e56d608d35ce135b975b5947b86ed3 (inspo from ritwickdey)

https://tutorial.tips/how-to-enable-disable-experimental-web-platform-features-chrome-browser/ (Face detection using Google Chrome) 
