#Modified code of Sentex Pygta5 2. train_model.py
#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np
import datetime
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

from keras import Input
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras import optimizers

BATCH = 500
lr = 0.0001 # Learning Rate

WIDTH = 400
HEIGHT = 300
EPOCHS = 1

MODEL_NAME = 'model_3_400x300_balanced_custom'

LOAD_MODEL = False

file_path = 'D:/Data Warehouse/pygta5/data/{}'.format(MODEL_NAME)
FILE_I_END = len(os.listdir(file_path))

if LOAD_MODEL:
    load_name = 'models/{}'.format(MODEL_NAME)
    model = load_model(load_name)
    print('We have loaded a previous model!')

# Define model
model = tf.keras.models.Sequential()
model.add(Conv2D(filters=2, kernel_size=2, input_shape=(HEIGHT,WIDTH,3)))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation = 'softmax'))

# Define model optimizer
adam = optimizers.Adam(lr = lr)

model.compile(
              optimizer = adam,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']
              )

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tensorboard = TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=True,
    update_freq='epochs', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)


model.summary()

# iterates through the training files
# for epoch in range(EPOCHS):
data_order = [i for i in range(0,FILE_I_END+1)]
shuffle(data_order)
for count,i in enumerate(data_order):

    try:
        file_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME, i)
        # full file info
        train_data = np.load(file_name, allow_pickle=True)

        SAMPLE = len(train_data)
        print('training_data-{}.npy - Sample Size: {}'.format(i,SAMPLE))

        X = np.array([i[0] for i in train_data]) / 255.0 # Divide to normalize values between 0 and 1
        print('X shape: {}'.format(str(X.shape)))
        Y = np.array([i[1] for i in train_data])

        # print("============================")
        # print("Epochs: {} - Steps: {}".format(epoch, count))

        model.fit(X, Y, epochs=EPOCHS, callbacks=[tensorboard_callback])
        # print("============================")

        if count%5 == 0 and count != 0:
            print('\nSAVING MODEL!\n')
            save_name = 'models/{}'.format(MODEL_NAME)
            model.save(save_name)

    except Exception as e:
        print(str(e))

print("FINISHED {} EPOCHS!".format(EPOCHS))
