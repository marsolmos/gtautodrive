import numpy as np

file_name = 'D:/Data Warehouse/pygta5/data/model_1_raw/training_data-1.npy'

train_data = np.load(file_name, allow_pickle=True)

# train_data[SAMPLE_NUMBER (0-499)][DIMENSION (0-1)]
print(train_data[499][0])
print('===============')
print(train_data[499][0].ndim)
print(train_data[499].shape)
