import numpy as np

file_name_1 = 'D:/Data Warehouse/pygta5/data/model_3_400x300_balanced_custom/training_data-1.npy'
train_data_1 = np.load(file_name_1, allow_pickle=True)

file_name_2 = 'D:/Data Warehouse/pygta5/data/model_3_400x300_balanced_custom/training_data-2.npy'
train_data_2 = np.load(file_name_2, allow_pickle=True)

file_name_3 = 'D:/Data Warehouse/pygta5/data/model_3_400x300_balanced_custom/training_data-3.npy'
train_data_3 = np.load(file_name_3, allow_pickle=True)

# train_data[SAMPLE_NUMBER (0-499)][DIMENSION (0-1)]
# print(train_data[499][0])
# print('===============')
# print(train_data[499][0].ndim)
# print(train_data[499].shape)
print(train_data_1.shape)

train_data_arr = []

train_data_arr.append(train_data_1)
# print(train_data_arr.shape)

train_data_arr.append(train_data_2)
# print(train_data_arr.shape)

train_data_arr.append(train_data_3)

data = np.concatenate(train_data_arr)
print(data.shape)
