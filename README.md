# Automated_Image_Captioning
Automated captioning of images using deep neural networks and MSCOCO dataset

The goal of this project is to train a deep neural network using MSCOCO dataset to generate captions for images.

The project is structured as a series of Jupyter notebooks:

**0_Dataset.ipynb :**

**1_Preliminaries.ipynb:**

  1. Loading and pre-processing of data from MSCOCO dataset
  2. Set up of encoder (pre-trained resnet-50) and decoder (LSTM)

**2_Training.ipynb:**

  1. Set hyperparameters based on initial tuning results
  2. Train the final layer of the encoder and the entire decoder

**3_Inference.ipynb :**

  1. Generate captions on some images after loading the trained model
