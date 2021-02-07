import time
import os
import numpy as np
import cv2

from config.collect_data.getkeys import key_check
from config.collect_data.grabscreen import grab_screen
from config.collect_data.preprocess_img import preprocess_img
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder
from config.collect_data.object_detector import download_object_detector
from config.collect_data.object_detector import detect_fn, get_distance


w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

starting_value = 1
MODEL_NAME = 'model_3_400x300_raw_custom'

while True:
    file_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME, starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        break


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output


def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    print('STARTING!!!')
    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,600))
            # Preprocess screen image
            temp_img,processed_img = preprocess_img(screen)

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([processed_img,output])

            # Display game image captured
            cv2.imshow('window', processed_img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            if len(training_data) % 100 == 0:
                print(len(training_data))

                if len(training_data) == 500:
                    np.save(file_name,training_data)
                    print('SAVED')
                    training_data = []
                    starting_value += 1
                    file_name = 'D:/Data Warehouse/pygta5/data/{}/training_data-{}.npy'.format(MODEL_NAME, starting_value)


        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main(file_name, starting_value)
