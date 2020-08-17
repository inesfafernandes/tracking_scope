# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:17:04 2020

@author: TeachinglabA
"""

import urllib

def download_import(filename):
    with open(filename, "wb") as f:
        # Downloading like that is needed because of Colab operating from a Google Drive folder that is just "shared with you".
        # https://drive.google.com/drive/folders/1U0xQMxVespjQilMhYW4mDxN02IwEW67I
        url = 'https://raw.githubusercontent.com/guillaume-chevalier/seq2seq-signal-prediction/master/{}'.format(filename)
        f.write(urllib.request.urlopen(url).read())

download_import("datasets.py")
download_import("plotting.py")
download_import("steps.py")

from typing import List
from logging import warning

import tensorflow as tf
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.metaopt.random import ValidationSplitWrapper
from neuraxle.metrics import MetricsWrapper
from neuraxle.pipeline import Pipeline, MiniBatchSequentialPipeline
from neuraxle.steps.data import EpochRepeater, DataShuffler
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.loop import ForEachDataInput
from sklearn.metrics import mean_squared_error
from tensorflow.python.client import device_lib
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import GRUCell, RNN, Dense
from tensorflow.python.training.adam import AdamOptimizer

from datasets import generate_data
from datasets import metric_3d_to_2d_wrapper
from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep
from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep
from plotting import plot_metrics
from steps import MeanStdNormalizer, ToNumpy, PlotPredictionsWrapper



def choose_tf_device():
    """
    Choose a TensorFlow device (e.g.: GPU if available) to compute on.
    """
    tf.debugging.set_log_device_placement(True)
    devices = [x.name for x in device_lib.list_local_devices()]
    print('You can use the following tf devices: {}'.format(devices))
    try:
        chosen_device = [d for d in devices if 'gpu' in d.lower()][0]
    except:
        warning(
            "No GPU device found. Please make sure to do `Runtime > Change Runtime Type` and select GPU for Python 3.")
        chosen_device = devices[0]
    print('Chosen Device: {}'.format(chosen_device))
    return chosen_device

chosen_device = choose_tf_device()