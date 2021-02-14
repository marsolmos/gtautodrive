import os
import numpy as np
from collections import Counter
from random import shuffle


"""
Extend, Balance & Append Data collected in step 1
================================
"""


def balance(MODEL_NAME_LOAD, MODEL_NAME_SAVE):
    '''
    Balance the extended data
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Balance extented data in MODEL_NAME_LOAD path and
    save it again in MODEL_NAME_LOAD folder
    '''

    print('\n\nBalancing data in {}\n'.format(MODEL_NAME_LOAD))

    file_path = 'D:/Data Warehouse/pygta5/data/{}'.format(MODEL_NAME_LOAD)
    FILE_I_END = len(os.listdir(file_path))

    for i in range(1,FILE_I_END+1):
        file_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME_LOAD, i)

        train_data = np.load(file_name, allow_pickle=True)

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

        w = w[:len(a)][:len(d)]
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
        print('Balanced file saved: ', i)

    return


def extend(MODEL_NAME_LOAD, MODEL_NAME_SAVE):
    '''
    Extend the existing data
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Duplicate existing data in MODEL_NAME_LOAD path and
    save it again in MODEL_NAME_LOAD folder
    '''

    print('\n\nExtending data in {}\n'.format(MODEL_NAME_LOAD))

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

    return


def append(MODEL_NAME_LOAD, MODEL_NAME_SAVE):
    '''
    Append data
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Append data MODEL_NAME_LOAD path into bigger files and
    save it again in MODEL_NAME_LOAD folder
    '''

    print('\n\nAppending data in {}\n'.format(MODEL_NAME_LOAD))

    file_path = 'D:/Data Warehouse/pygta5/data/{}'.format(MODEL_NAME_LOAD)
    FILE_I_END = len(os.listdir(file_path))

    train_data_arr = []
    filenum = 1

    for i in range(1,FILE_I_END+1):

        # Find the path of the file, load it and append it
        filepath = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME_LOAD, i)
        file = np.load(filepath, allow_pickle=True)
        print('Reading File {}'.format(i))
        # If file is empty, then pass
        if file.ndim != 2:
            pass
        else:
            train_data_arr.append(file)

        # Delete file once appended in train_data_arr
        os.remove(filepath)

        if i % 25 == 0:
            SAVE_PATH ="D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy".format(MODEL_NAME_SAVE, filenum)
            data = np.concatenate(train_data_arr)
            np.save(SAVE_PATH,data)
            print('SAVED!\n\n')
            filenum += 1
            train_data_arr = []

    SAVE_PATH ="D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy".format(MODEL_NAME_SAVE, filenum)
    print('SAVING LAST GROUP OF DATA')
    data = np.concatenate(train_data_arr)
    np.save(SAVE_PATH,data)
    print('SAVED!\n\n')

    return


raw = 'model_4_400x300_raw_inceptionv3' # Load data from this path
balanced = 'model_4_400x300_balanced_custom' # Save data in this path

balance(raw, balanced)
extend(balanced, balanced)
append(balanced, balanced)
