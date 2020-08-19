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

#Now the WindowGenerator object gives you access to the tf.data.Dataset objects, so you can easily iterate over the data.

# Each element is an (inputs, label) pair
w2.train.element_spec #this command tells you the structure, dtypes and shapes of the dataset elements.

for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
#%% Single step model

#The simplest model you can build is one that predicts a single feature's
# value, 1 timestep (1h) in the future based only on the current conditions.

single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['T (degC)'])
single_step_window

#The window object creates tf.data.Datasets from the training, validation,
# and test sets, allowing you to easily iterate over batches of data

for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
#This first task is to predict temperature 1h in the future given the 
#current value of all features. The current values include the current 
#temperature.

#we'll start with a model that just returns the current temperature as 
#the prediction, predicting "No change". This is a reasonable baseline 
#since temperature changes slowly. Of course, this baseline will work 
#less well if you make a prediction further in the future

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__() #gives you access to methods in a superclass from the subclass that inherits from it, in this case we inherit from tf.keras.Model class
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]


#Instantiate and evaluate this model, this is the baseline, our "naive model"

baseline = Baseline(label_index=column_indices['T (degC)'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

#The wide_window doesn't change the way the model operates. The model 
#still makes predictions 1h into the future based on a single input time 
#step. Here the time axis acts like the batch axis: Each prediction is 
#made independently with no interaction between time steps.

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['T (degC)'])

wide_window

#This expanded window can be passed directly to the same baseline model. 
#This is possible because the inputs and labels have the same number of 
#timesteps, and the baseline just forwards the input to the output:
    
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', baseline(single_step_window.example[0]).shape)

#Plotting the baseline model's predictions you can see that it is simply
# the labels, shifted right by 1h.

wide_window.plot(baseline)

#Linear model
#The simplest trainable model you can apply to this task is to insert 
#linear transformation between the input and output. In this case the 
#output from a time step only depends on that step:
    
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)

#Since in this tutorial we will train many models, will package the training procedure into a function:

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
    #Stop training when a monitored metric has stopped improving.
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience, #Number of epochs with no improvement after which training will be stopped.
                                                    mode='min') #In min mode, training will stop when the quantity monitored has stopped decreasing

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

#Train the model and evaluate its performance

history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

#Like the baseline model, the linear model can be called on batches of
# wide windows. Used this way the model makes a set of independent 
#predictions on consecuitive time steps. The time axis acts like another batch axis.

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', linear(wide_window.example[0]).shape)

#Here is the plot of its example predictions on the wide_widow

wide_window.plot(linear)

#One advantage to linear models is that they're relatively simple to 
#interpret. we can pull out the layer's weights, and see the weight 
#assigned to each input:

plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)

#Sometimes the model doesn't even place the most weight on the input T 
#(degC). This is one of the risks of random initialization.

#Dense
#stacks several a few Dense layers between the input and the output:

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

wide_window.plot(dense)

plt.bar(x = range(len(train_df.columns)),
        height=dense.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)

#Multi-step dense

#A single-time-step model has no context for the current values of its 
#inputs. It can't see how the input features are changing over time. To 
#address this issue the model needs access to multiple time steps when 
#making predictions

#The baseline, linear and dense models handled each time step independently. 
#Here the model will take multiple time steps as input to produce a single 
#output

#Creating a WindowGenerator that will produce batches of the 3h of inputs
# and, 1h of labels

CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['T (degC)'])

conv_window

conv_window.plot()
plt.title("Given 3h as input, predict 1h into the future.")

# we can train a dense model on a multiple-input-step window by adding a
# layers.Flatten as the first layer of the model

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]), #Layer that reshapes inputs into the given shape.
])

print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

#takes three time steps (from all features) and outputs information about
# the next time step only (of the temperature feature)

history = compile_and_fit(multi_step_dense, conv_window)

IPython.display.clear_output() # ask afonso
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

conv_window.plot(multi_step_dense)

#The main down-side of this approach is that the resulting model can only
# be executed on input wndows of exactly this 

print('Input shape:', wide_window.example[0].shape)
try:
  print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
  print(f'\n{type(e).__name__}:{e}')
  
# The convolutional models in the next section fix this problem.

#Convolutional neural network

#A convolution layer (layers.Conv1D) also takes multiple time steps as 
#input to each prediction.

conv_model = tf.keras.Sequential([
    #The layers.Flatten and the first layers.Dense are replaced by a layers.Conv1D.
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),#The layers.Reshape is no longer necessary since the convolution keeps the time axis in its output.
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

#training and evaluating performance
history = compile_and_fit(conv_model, conv_window)

IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

#the performance is similar to the multi_step_dense

#The difference between this conv_model and the multi_step_dense model is
# that the conv_model can be run on inputs of any length. The convolutional 
#layer is applied to a sliding window of inputs

#If you run it on wider input, it produces wider output

print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)

#Note that the output is shorter than the input. To make training or 
#plotting work, you need the labels, and prediction to have the same 
#length. So build a WindowGenerator to produce wide windows with a few 
#extra input time steps so the label and prediction lengths match:
    
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['T (degC)'])

wide_conv_window

print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

# Note the 3 input time steps before the first prediction. Every 
#prediction here is based on the 3 preceding timesteps:

wide_conv_window.plot(conv_model)

#Recurrent neural network
#In this case a layer called Long Short Term Memory will be used

#An important constructor argument for all keras RNN layers is the 
#return_sequences argument. This setting can configure the layer in one 
#of two ways.
    #-If False, the default, the layer only returns the output of the 
    #final timestep, giving the model time to warm up its internal state
    #before making a single prediction:
    #-If True the layer returns an output for each input. 

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

#With return_sequences=True the model can be trained on 24h of data at a
# time.
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)

history = compile_and_fit(lstm_model, wide_window)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

wide_window.plot(lstm_model)

#comparing the models performance

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()

#printing the name of the model and the associated performance (in this
# case mean square error the lower the better)
for name, value in performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')
  
#Multi-output models

#The models so far all predicted a single output feature, T (degC), for a
# single time step.

#All of these models can be converted to predict multiple features just 
#by changing the number of units in the output layer and adjusting the 
#training windows to include all features in the labels. 

single_step_window = WindowGenerator(
    # `WindowGenerator` returns all features as labels if you 
    # don't set the `label_columns` argument.
    input_width=1, label_width=1, shift=1)

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1)

for example_inputs, example_labels in wide_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
#baseline: The same baseline model can be used here, but this time 
#repeating all features instead of selecting a specific label_index.

baseline = Baseline()
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(wide_window.val)
performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)

#dense
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features)
])

history = compile_and_fit(dense, single_step_window)

IPython.display.clear_output()
val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

#RNN

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1)

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

history = compile_and_fit(lstm_model, wide_window)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate( wide_window.val)
performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0)

#Residuals connections

#Every model trained in this tutorial so far was randomly initialized, 
#and then had to learn that the output is a a small change from the 
#previous time step.
#It's common in time series analysis to build models that instead of 
#predicting the next value, predict the how the value will change in the
# next timestep. Similarly, "Residual networks" or "ResNets" in deep 
#learning refer to architectures where each layer adds to the model's 
#accumulating result.

#Essentially this initializes the model to match the Baseline. For this 
#task it helps models converge faster, with slightly better performance.It
#can be used in conjunction with any model discussed in this tutorial.

class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)

    # The prediction for each timestep is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return inputs + delta

residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small
        # So initialize the output layer with zeros
        kernel_initializer=tf.initializers.zeros)
]))

history = compile_and_fit(residual_lstm, wide_window)

IPython.display.clear_output()
val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)


# Overall performance for the multi-output models

x = np.arange(len(performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
plt.ylabel('MAE (average over all outputs)')
_ = plt.legend()

for name, value in performance.items():
  print(f'{name:15s}: {value[1]:0.4f}')
  
#%% Multi-step models

#This section looks at how to expand the previous models to make multiple
# time step predictions.

#Thus, unlike a single step model, where only a single future point is 
#predicted, a multi-step model predicts a sequence of the future values.

#the models will learn to predict 24h of the future, given 24h of the past.
OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
multi_window


#A simple baseline for this task is to repeat the last input time step 
#for the required number of output timesteps:
    
class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.val, verbose=0)
multi_window.plot(last_baseline)

#Another simple approach is to repeat the previous day, assuming tomorrow
# will be similar:
    
class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(repeat_baseline)

#single shot

#the model makes the entire sequence prediction in a single step.

#linear
multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_linear_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_linear_model)

#dense
#Adding a layers.Dense between the input and output gives the linear 
#model more power, but is still only based on a single input timestep

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_dense_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_dense_model)

#CNN
#A convolutional model makes predictions based on a fixed-width history,
# which may lead to better performance than the dense model since it can
# see how things are changing over time:
    
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model)

#RNN
#can learn to use a long history of inputs, if it's relevant to the 
#predictions the model is making. Here the model will accumulate internal
# state for 24h, before making a single prediction for the next 24h.

#In this single-shot format, the LSTM only needs to produce an output at
# the last time step, so return_sequences=False.

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.train, verbose=0)
multi_window.plot(multi_lstm_model)

#%% Advanced: Autoregressive model

#In some cases it may be helpful for the model to decompose this 
#prediction into individual time steps. Then each model's output can be
# fed back into itself at each step and predictions can be made 
#conditioned on the previous one, like in the classic Generating 
#Sequences With Recurrent Neural Networks.

#RNN
#This tutorial only builds an autoregressive RNN model, but this pattern
# could be applied to any model that was designed to output a single 
#timestep.

class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)
    
feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

#The first method this model needs is a warmup method to initialize its
# internal state based on the inputs. Once trained this state will 
#capture the relevant parts of the input history. This is equivalent to
# the single-step LSTM model from earlier

def warmup(self, inputs):#ASK AFONSO aqui nao e preciso fzr property cm antes?
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs) #ASK AFONSO

  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup

prediction, state = feedback_model.warmup(multi_window.example[0])
prediction.shape

def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the lstm state
  prediction, state = self.warmup(inputs)

  # Insert the first prediction
  predictions.append(prediction)

  # Run the rest of the prediction steps
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output
    predictions.append(prediction)

  # predictions.shape => (time, batch, features)
  #NOTE: Stacking a python list like this only works with eager-execution,
  #using Model.compile(..., run_eagerly=True) for training, or with a 
  #fixed length output. For a dynamic output length you would need to 
  #use a tf.TensorArray instead of a python list, and tf.range instead 
  #of the python range.
  predictions = tf.stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = tf.transpose(predictions, [1, 0, 2])
  return predictions

FeedBack.call = call

print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)

history = compile_and_fit(feedback_model, multi_window)

IPython.display.clear_output()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(feedback_model)

#comparing performances

x = np.arange(len(multi_performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()

for name, value in multi_performance.items():
  print(f'{name:8s}: {value[1]:0.4f}')