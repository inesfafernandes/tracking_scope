# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:53:14 2020

@author: Inês Fernandes
"""

#Tutorial time series forecasting

#%% importing libraries needed

import os #os module provides functions for interacting with the operating system
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#%% downloading data

#dataset contains 14 different features such as air temperature, atmospheric pressure, and humidity, the 15th colum is the date time
#These were collected every 10 minutes

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path,_ = os.path.splitext(zip_path) #used to split the path name into a pair root and ext (ext stands for extension, in this case csv); if we dont want the variable we use _ ?

#This tutorial will just deal with hourly predictions, so we start by sub-sampling the data from 10 minute intervals to 1h

df = pd.read_csv(csv_path)
# starting from index 5 take every 6th record.
df = df[5::6]
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S') #helps to convert string Date time into Python Date time object; removes the item at the given index from the list and returns the removed item, in this case we are extracting he values from colum date time
df.head() # shows the first 5 rows of the 14 features

#%% Evolution of a few features over time.

plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)'] #names of features
plot_features = df[plot_cols]# matrix containing values of the 3 vectors (one for each feature)
plot_features.index = date_time #returns the index of the specified element in the list.
_ = plot_features.plot(subplots=True)

#to zoom in
plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)

#%% Inspect and cleanup

df.describe().transpose() #.describe is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values; .transpose permutes the axes of an array (switched the features with the statistics)

#given what we absorve 

wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame
df['wv (m/s)'].min()

#%% Feature engineering

#The last column of the data, wd (deg), gives the wind direction in units of degrees. Angles do not make good model inputs, 360° and 0° should be close to each other, and wrap around smoothly
#current distribution of wind data

plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400) #Make a 2D histogram plot. xaxis is wind direction yaxis is the wind velocity, number of bins in each dimension (nx, ny = bins))
plt.colorbar() #show olorbar
plt.xlabel('Wind Direction [deg]')
plt.ylabel('Wind Velocity [m/s]')

#Direction shouldn't matter if the wind is not blowing. this will be easier for the model to interpret if you convert the wind direction and velocity columns to a wind vector

wv = df.pop('wv (m/s)') #extract wv(m/s) from data set to variable wv
max_wv = df.pop('max. wv (m/s)') #extract max.wv(m/s) from data set to variable max_wv

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

#plotting new wind vector arrangement

plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [m/s]')
plt.ylabel('Wind Y [m/s]')
ax = plt.gca()
ax.axis('tight')

# Date Time column is very useful, but not in this string form. Start by converting it to seconds

timestamp_s = date_time.map(datetime.datetime.timestamp)

#Being weather data it has clear daily and yearly periodicity. There are many ways you could deal with periodicity.
#A simple approach to convert it to a usable signal is to use sin and cos to convert the time to clear "Time of day" and "Time of year" signals

day = 24*60*60 # 1 day in sec
year = (365.2425)*day # 1 year in sec

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')

#This gives the model access to the most important frequency features. 
#In this case we knew ahead of time which frequencies were important, if we didnt we could figure it out by doing an fft

fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft)) #Return evenly spaced values within a given interval

n_samples_h = len(df['T (degC)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)

f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
_ = plt.xlabel('Frequency (log scale)')
#obvious peaks at frequencies near 1/year and 1/day

#%%Spliting the data

#We'll use a (70%, 20%, 10%) split for the training, validation, and test sets

column_indices = {name: i for i, name in enumerate(df.columns)} #creating a dictionary with name of features and column indice
#same as
#column_indices = {}
#for i, name in enumerate(df.columns):
    #column_indices[name] = i


n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

#%% Normalizing data

#It is important to scale features before training a neural network. Normalization is a common way of doing this scaling. Subtract the mean and divide by the standard deviation of each feature.

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std #normalizing training set
val_df = (val_df - train_mean) / train_std #normalizing validation set
test_df = (test_df - train_mean) / train_std #normalizing test set

#taking a look at the distribution of the features
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

#%% Data windowing

#The models in this tutorial will make a set of predictions based on a window of consecutive samples from the data.
#This tutorial builds a variety of models (including Linear, DNN, CNN and RNN models), and uses them for both:
    #Single-output, and multi-output predictions.
    #Single-time-step and multi-time-step predictions.

#creating window generator class

class WindowGenerator():
    
    # init method or constructor 
    #self represents the instance of the class
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

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
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

# window for single prediction 24h into the future, given 24h of history
w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['T (degC)'])
w1

#prediction 1h into the future, given 6h of history
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['T (degC)'])
w2

#SPLIT
#Given a list consecutive inputs, the split_window method will convert them to a window of inputs and a window of labels.

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
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')
#The code above took a batch of 2, 7-timestep windows, with 19 features at each time step. It split them into a batch of 6-timestep, 19 feature inputs, and a 1-timestep 1-feature label.

#plot to visualize the windows
w2.example = example_inputs, example_labels

def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
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

#for other column
w2.plot(plot_col='p (mbar)')

#%% Create tf.data.Datasets

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  # Creates a dataset of sliding windows over a timeseries provided as array.
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)
  
  ## returns a map object(which is an iterator) of the results after applying the given function to each item of a given iterable (list, tuple etc.)
  ds = ds.map(self.split_window)

  return ds
    
WindowGenerator.make_dataset = make_dataset


#standard example batch for easy access and plotting

@property # ask afonso
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None) #returns the value of the named attribute of an object
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

#continue in 4. Create tf.data.Datasets