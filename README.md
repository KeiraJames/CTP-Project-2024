# <p align="center"> AI and REAL Image Classification </p> 
## Introduction
An image classification model that takes an image as an input, and predicts whether the image is real or created by A.I. based on the results from two models.

## Models Developed/Implemented
1. A Convolutional Neural Network (CNN) architecture was developed by our team
2. An EfficientNet-based model architecture implemented by our team.

## Team 
|Names                  |Primary Roles                                | Secondary Roles                                   |
| --------------------- | ------------------------------------------- | ------------------------------------------------- |
| Farhikhta Farzan      | Data collector and cleaner                  | Model Interpreation/ Visualization and deployment |
| Keira James           | Feature Engineer                            | Model Developer                                   |
| Tesneem Essa          | Feature Engineer                            | Model Developer                                   |

## Deployed Version
Try the project here 🎨 !

https://huggingface.co/spaces/Digital-Detectives/AI-vs-Real-Image-Detection

## Project Outline
The goal of this project is to develop a deep learning model that can accurately distinguish between real images and AI-generated images. We will collect datasets of real images and fake images. The data will be preprocessed, normalized, and augmented to enhance training. Using TensorFlow and Keras, we will design a Convolutional Neural Network (CNN) for classification, and validating performance through a confusion matrix. Finally, the project will include documentation of the process, findings, and suggestions for future improvements.

## Datasets Used
[CIFAKE: Real and AI-Generated Synthetic Images]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)) <br>
CIFAKE is a dataset that contains 60,000 synthetically-generated images and 60,000 real images (collected from CIFAR-10). The dataset contains two classes - REAL and FAKE.
For REAL, images are collected from Krizhevsky & Hinton's CIFAR-10 dataset. For the FAKE images, the equivalent of CIFAR-10 with Stable Diffusion version 1.4 was generated.
There are 100,000 images for training (50k per class) and 20,000 for testing (10k per class)

[Paintings from 10 different popular artists]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/binukandagedon/paintings-from-10-different-popular-artists)) <br>
This data set is about paintings from 10 different artists. The artists are davinci, frida kahlo, henri matisse, jackson pollock, johannes vermeer, picasso, piere auguste, raphael, rembrandt, and van gough

[ArtStyles-Dataset]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/norasami/artstylesdataset/data)) <br>
ArtStyles dataset contains 360 images containing 3 different digital art styles. The art style includes Anime, Comic and Semi Realism

[DALLE Art March, 2023]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/nikbearbrown/dalle-art-march-2023))<br>
These are AI-generated art images using Midjourney for the month of March 2023. It contains around 660 files

[Detecting AI-generated Artwork]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/birdy654/detecting-ai-generated-artwork))<br>
This dataset was produced as part of the study "AI Generated Art: Latent Diffusion-Based Style and Detection". It contains 1705 fake images and 1705 real images.

[Midjourney Images & Prompt]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/cyanex1702/midjourney-imagesprompt))<br>
This dataset is an extensive collection of pictures produced by using the mid-journey idea. It provides a wide range of varied photos with corresponding prompts that are created automatically by means of an advanced captioning system.This dataset, designed primarily for diffusion model training, is a valuable resource for improving machine learning capabilities in image generation. Using the picture and accompanying cues, researchers can delve into the complexities of training stable diffusion models capable of producing visuals resembling mid-journey scenarios.

[BEAR x DALLE - Robot Illustrations]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/nikbearbrown/bear-x-dalle-robot-illustrations))<br>
This is a dataset for illustrations of robots in the styles of well-known artists, art genre styles, perspectives, formats, and lighting.
This is a dataset of illustrations of robots in the styles of well-known artists, art genre styles, perspective, formats, and lighting, as part of the BEAR x DALLE project, which focuses on computational art. 

[Generated Abstract Art Gallery]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/bryanb/generated-abstract-dataset-diffusion))<br>
This dataset used Generative x Diffusion models to generate 512x512 AI images with abtract style art.

[Anime Chibi Datasets]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/hirunkulphimsiri/anime-chibi-datasets))<br>
Anime Chibi Characters Datasets : Scraped From safebooru.org/Scraped with tags "chibi standing white_background solo -translation_request -text"
Aims to be used in gans to generate Chibi characters

[Cats and Dogs Cartoons]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/maroquio/cats-and-dogs-cartoons))<br>
This is a great dataset for learning image classification. The dataset has 400 images (200 cats and 200 dogs) in hand drawn cartoon style. Each image is 1024x1024 pixels. The data was generated in MidJourney and was proved to be very useful in my Machine Learning classes. 

[Indian Paintings Dataset]([/guides/content/editing-an-existing-page](https://www.kaggle.com/datasets/ajg117/indian-paintings-dataset?select=gond+painting))<br>
Dive into the vibrant kaleidoscope of Indian art with our meticulously curated Indian Painting Styles Dataset! Number of classes - 8 (gond, kalighat, kangra, kerala mural, madhubani, mandana, pichwai, warli paintings) Total Images - 2249 images. File Formats - .jpg, .jpeg, .png, .webp

## Technologies Used
1. Kaggle
2. Numpy
3. Panda
4. cv2
5. Pillow
6. Seaborn
7. Tensorflow/Keras
8. Scikit-Learn
9. Streamlit

## Model Performance
CNN Model:
      
      Accuracy: 97.62%

      
Efficiencynet Model:

      Accuracy: 97.72%
      Precision: 97.4%
      Recall : 98.05%
      F1 score: 97.72%


## UI

https://github.com/user-attachments/assets/62cb9dd8-0a9e-443f-a1dd-f90f2e0ce4fd



## Practical Application
Our project is crucial in today’s world, where AI-generated content is increasingly prevalent. This model can be used by social media platforms, news organizations, and even everyday people to verify the authenticity of images, helping to fight against misinformation and ensure the integrity of visual media.

## Set up
1. Navigate to desired library
```
  cd your_directory
```
2. Clone repository
```
  git clone https://github.com/KeiraJames/CTP-Project-2024.git
```
3. Navigate to repo
```
 cd CTP-Project-2024
```
4. Create virtual environment
```
 python -m venv .aivsreal-venv
```
5. Activate virtual environment (MAC)
```
 source .aivsreal/bin/activate
```
6. Install the requirements
 ```
   pip install -r requirements.txt
 ```
7. Run streamlit
 ```
   streamlit run app.py
 ```






