# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:09:50 2020

@author: TeachinglabA

Predicting sinusoidal waves
"""
#%%Loading required libraries

import math
import numpy as np
import matplotlib.pyplot as plt

import random

import os #os module provides functions for interacting with the operating system
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import tensorflow as tf

#%% Creating data

n_samples=50
sequence_length=3000

batch_data = []

for _ in range(n_samples):
    rand = random.random() * 2 * math.pi #random.random returns the next random floating point number in the range [0.0, 1.0).

    sig1 = np.sin(np.linspace(rand, 20.0 * math.pi + rand, sequence_length * 2)) 
    #np.linspace(start,stop,num); Returns evenly spaced numbers over a specified interval. start: starting value of the sequence. Stop: end value of the sequence, num:Number of samples to generate
    data= sig1.T #transposes de arrays

    batch_data.append(sig1) #append is used in list
     

batch_data = np.array(batch_data) #converted to an array
#data_sin=batch_data.T
data_sin=pd.DataFrame(data=batch_data.T)
data_sin.head()


# shape: (n_samples, seq_length, output_dim)

#%% exploring the data

for i in range(6): # we want to do this for the first 25 figures
    plt.subplot(2,3,i+1)#like in matlab we say we want to plot a 5 by 5 image and the last number corrends to the window of the current plot, sense python indices start at 0 we use i+1
    plt.plot(data_sin[:,i])# shows image and sets colormap to binary
plt.show() #shows figure w all plots


#%% splitting the data

n = len(data_sin) #sequence length
train_data = data_sin[0:int(n*0.7)]
val_data = data_sin[int(n*0.7):int(n*0.9)]
test_data= data_sin[int(n*0.9):]

#%% Creating window

class WindowGenerator():
    
    # init method or constructor 
    #self represents the instance of the class
  def __init__(self, input_width, label_width, shift,
               train_x=train_data, val_x=val_data, test_x=test_data,label_columns=None):
    # Store the raw data.
    self.train_x = train_x
    self.val_x = val_x
    self.test_x = test_x
    
    # Work out the label column indices.
    
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {i: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {i: i for i, name in
                           enumerate(data_sin.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width) #used to slice a given sequence; 0 is the starting integer where the slicing of the object starts input_width is the integer until which the slicing takes place
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',f'Label column name(s): {self.label_columns}'])

# window for single prediction 20 time steps into the future, given 50 time steps of history
w1 = WindowGenerator(input_width=50, label_width=1, shift=20,label_columns=[1])
w1

#prediction 1 time step into the future, given 20 time steps of history of history
w2 = WindowGenerator(input_width=20, label_width=1, shift=1,label_columns=[1])
w2

#%% Split inputs and labels

#Given a list of consecutive inputs, the split_window method will convert them to a window of inputs and a window of labels.

def split_window(self, nsamples):
  inputs = nsamples[:, self.input_slice, :]
  labels = nsamples[:, self.labels_slice, :]
  if self.label_columns is not None:
      #Stacks a list of rank-R tensors into one rank-(R+1) tensor.
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

#result
# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_data[:w2.total_window_size]),
                           np.array(train_data[100:100+w2.total_window_size]),
                           np.array(train_data[200:200+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')


