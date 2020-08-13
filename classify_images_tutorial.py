# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:54:03 2020

@author: Ines Fernandes
"""
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#%% import fashion dataset

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] #to substitute the labels that are just numbers

#%%exploring data

train_images.shape # shows size like in the variable explorer
len(train_labels) # shows length of the vaiable
train_labels # shows whats stored in the variable and its type

#%% preprocesing data

plt.figure() #opens figure
plt.imshow(train_images[0]) # plots the image in indice 0 of train images array
plt.colorbar() # plots a color bar
plt.show()

# pixel values of the image range form 0 to 255 we want to values to be between 0 and 1 (why?)

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10)) # opens a figure with a specific size
for i in range(30): # we want to do this for the first 25 figures
    plt.subplot(6,5,i+1)#like in matlab we say we want to plot a 5 by 5 image and the last number corrends to the window of the current plot, sense python indices start at 0 we use i+1
    plt.xticks([])#Get or set the current tick locations and labels of the x-axis.
    plt.yticks([])#Get or set the current tick locations and labels of the y-axis.
    plt.imshow(train_images[i], cmap=plt.cm.binary)# shows image and sets colormap to binary
    plt.xlabel(class_names[train_labels[i]]) #sets the x label as the fashion item label, the class names order much the numbers in labels so we get train label in the indice of the number of the figure and then that label is used as an indice to slect a class name
plt.show() #shows figure w all plots

#%% Building the model

#setting up the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
    keras.layers.Dense(128, activation='relu'),# first dense layer, it has 128 nodes. The activation function is responsible for transforming the summed weighted input from the node into the activation of the node or output for that input. The rectified linear unit (ReLU) activation function is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero
    keras.layers.Dense(10)# second dense layer returns logits(?) arrays with length of 10
])
# Each node contains a score that indicates the current image belongs to one of the 10 classes.

#%% Compiling the model

#Loss function —This measures how accurate the model is during 
#training. You want to minimize this function to "steer" the model
# in the right direction.

#Optimizer —This is how the model is updated based on the data it
# sees and its loss function.

#Metrics —Used to monitor the training and testing steps. The 
#following example uses accuracy, the fraction of the images that
# are correctly classified.

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
#well suited for problems that are large in terms of data/parameters".

#move to train the model