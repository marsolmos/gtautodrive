#Modified code of Sentex Pygta5 2. train_model.py
#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import cv2
import time
import os
import pandas as pd
from collections import deque
from random import shuffle
import pickle

# Do not display following messages:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import Input
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import TensorBoard
from keras.models import load_model

BATCH = 16

WIDTH = 80
HEIGHT = 60
EPOCHS = 30

MODEL_NAME = 'model_1_balanced'

LOAD_MODEL = False

file_path = 'D:/Data Warehouse/pygta5/data/{}'.format(MODEL_NAME)
FILE_I_END = len(os.listdir(file_path))

# Define training model
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Dense(input_shape=(HEIGHT, WIDTH, 3), activation='relu'),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(9, activation='relu')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

input_tensor = Input(shape=(WIDTH,HEIGHT,3))
model = InceptionV3(
                    include_top=True,
                    input_tensor=input_tensor,
                    pooling='max',
                    classes=9,
                    weights=None)
model.compile('Adagrad', 'categorical_crossentropy')

"""
tensorboard = TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=True,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)
"""

if LOAD_MODEL:
    load_name = 'models/{}'.format(MODEL_NAME)
    model = load_model(load_name)
    print('We have loaded a previous model!')

# iterates through the training files
for e in range(EPOCHS):
    data_order = [i for i in range(0,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):

        try:
            file_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME, i)
            # full file info
            train_data = np.load(file_name, allow_pickle=True)

            SAMPLE = len(train_data)
            print('training_data-{}.npy - Sample Size: {} - Batch Size: {}'.format(i,SAMPLE,BATCH))

            X = np.array([i[0] for i in train_data]).reshape(-1,WIDTH,HEIGHT,3) #Pre reshaped at recording
            Y = np.array([i[1] for i in train_data])

            print("============================")
            print("Epochs: {} - Steps: {}".format(e, count))
            model.fit(X, Y, batch_size=BATCH ,epochs=1, validation_split=0.02) #, callbacks=[tensorboard])
            print("============================")

            if count%5 == 0 and count != 0:
                print('SAVING MODEL!')
                save_name = 'models/{}'.format(MODEL_NAME)
                model.save(save_name)

        except Exception as e:
            print(str(e))

print("FINISHED {} EPOCHS!".format(EPOCHS))
