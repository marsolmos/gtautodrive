import os
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

MODEL_NAME_LOAD = 'model_2_400x300_raw' # Load raw data from this path
MODEL_NAME_SAVE = 'model_2_400x300_balanced' # Save balanced data in this path

file_path = 'D:/Data Warehouse/pygta5/data/{}'.format(MODEL_NAME_LOAD)
FILE_I_END = len(os.listdir(file_path))

for i in range(1,FILE_I_END+1):
    file_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME_LOAD, i)

    train_data = np.load(file_name, allow_pickle=True)

    df = pd.DataFrame(train_data)

    w = []
    s = []
    a = []
    d = []
    wa = []
    wd = []
    sa = []
    sd = []
    nk = []

    shuffle(train_data)

    for data in train_data:
        img = data[0]
        choice = data[1]

        if choice == [1,0,0,0,0,0,0,0,0]:
            w.append([img,choice])
        elif choice == [0,1,0,0,0,0,0,0,0]:
            s.append([img,choice])
        elif choice == [0,0,1,0,0,0,0,0,0]:
            a.append([img,choice])
        elif choice == [0,0,0,1,0,0,0,0,0]:
            d.append([img,choice])
        elif choice == [0,0,0,0,1,0,0,0,0]:
            wa.append([img,choice])
        elif choice == [0,0,0,0,0,1,0,0,0]:
            wd.append([img,choice])
        elif choice == [0,0,0,0,0,0,1,0,0]:
            sa.append([img,choice])
        elif choice == [0,0,0,0,0,0,0,1,0]:
            sd.append([img,choice])
        elif choice == [0,0,0,0,0,0,0,0,1]:
            nk.append([img,choice])
        else:
            print('no matches')

    w = w[:len(wa)][:len(wd)]
    s = s[:len(w)]
    a = a[:len(w)]
    d = d[:len(w)]
    wa = wa[:len(w)]
    wd = wd[:len(w)]
    sa = sa[:len(w)]
    sd = sd[:len(w)]
    nk = nk[:len(w)]

    final_data = w + s + a + d + wa + wd + sa + sd + nk
    shuffle(final_data)

    save_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME_SAVE, i)
    np.save(save_name, final_data)
    print('File saved: ', i)
