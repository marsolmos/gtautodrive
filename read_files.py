import numpy as np

file_name = 'D:/Data Warehouse/pygta5/data/model_1_balanced/training_data-1.npy'

train_data = np.load(file_name, allow_pickle=True)

print(train_data.shape)
print(train_data[1][0].shape)

file_name = 'D:/Data Warehouse/pygta5/data/model_1_balanced/training_data-7.npy'

train_data = np.load(file_name, allow_pickle=True)

# print(train_data.shape)



file_name = 'D:/Data Warehouse/pygta5/data/model_1_raw/training_data-1.npy'

train_data = np.load(file_name, allow_pickle=True)

# print(train_data.shape)

file_name = 'D:/Data Warehouse/pygta5/data/model_1_raw/training_data-7.npy'

train_data = np.load(file_name, allow_pickle=True)

# print(train_data.shape)
