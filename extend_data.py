import os
import numpy as np
import pandas as pd

MODEL_NAME_LOAD = 'model_3_400x300_raw_custom' # Load raw data from this path
MODEL_NAME_SAVE = 'model_3_400x300_raw_custom' # Save balanced data in this path

file_path = 'D:/Data Warehouse/pygta5/data/{}'.format(MODEL_NAME_LOAD)
FILE_I_END = len(os.listdir(file_path))
NEW_INDEX_SAVE = FILE_I_END

for i in range(1,FILE_I_END+1):
    NEW_INDEX_SAVE += 1

    file_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME_LOAD, i)
    train_data = np.load(file_name, allow_pickle=True)
    save_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME_SAVE, NEW_INDEX_SAVE)
    np.save(save_name, train_data)

    print('File saved --> Original index: {}  -  New index: {} '.format(i, NEW_INDEX_SAVE))
