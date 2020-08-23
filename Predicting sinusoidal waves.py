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

date_time=np.arange(0,6000,1)
plot_cols = [1, 45, 3]
plot_features = data_sin[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

#%% splitting the data

column_indices = {i: i for i, name in enumerate(data_sin.columns)}

n = len(data_sin) #sequence length
train_data = data_sin[0:int(n*0.7)]
val_data = data_sin[int(n*0.7):int(n*0.9)]
test_data= data_sin[int(n*0.9):]

num_features = data_sin.shape[1]

#%% normalizing data (does it make sense?)

train_mean = train_data.mean()
train_std = train_data.std()

train_data = (train_data - train_mean) / train_std
val_data = (val_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std

data_std = (data_sin - train_mean) / train_std
data_std = data_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=data_std)
_ = ax.set_xticklabels(data_sin.keys(), rotation=90)

#%% Creating window

class WindowGenerator():
    
    # init method or constructor 
    #self represents the instance of the class
  def __init__(self, input_width, label_width, shift,
               train_data=train_data, val_data=val_data, test_data=test_data,label_columns=None):
    # Store the raw data.
    self.train_data = train_data
    self.val_data = val_data
    self.test_data = test_data
    
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

#prediction 1 time step into the future, given 6 time steps of history of history
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,label_columns=[1])
w2

#%% Split inputs and labels

#Given a list of consecutive inputs, the split_window method will convert them to a window of inputs and a window of labels.

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
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

#%% Plot method

w2.example = example_inputs, example_labels

def plot(self, model=None, plot_col=1, max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [h]')

WindowGenerator.plot = plot

w2.plot()


#%%

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_x)

@property
def val(self):
  return self.make_dataset(self.val_x)

@property
def test(self):
  return self.make_dataset(self.test_x)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Each element is an (inputs, label) pair
for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
#%% Compile and fit model

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

#%% Wide window

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=[1])

wide_window

#%% baseline
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=[1])
single_step_window

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices[1])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

wide_window.plot(baseline)
#%% CNN

CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=[1])

conv_window

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

history = compile_and_fit(conv_model, conv_window)

val_performance = {}
performance = {}
IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)

# Since the output is shorter than the input. To make training or plotting work, we
# need the labels, and prediction to have the same length. So we need to build a 
#WindowGenerator to produce wide windows with a few extra input time steps so the 
#label and prediction lengths match

#%% Wide conv window

LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=[1])

wide_conv_window

print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

wide_conv_window.plot(conv_model)