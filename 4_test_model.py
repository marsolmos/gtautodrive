import os
import numpy as np
import cv2
import time
from collections import deque, Counter
import random
from statistics import mode,mean
import numpy as np

from config.collect_data.grabscreen import grab_screen
from config.collect_data.vehicle_detector import vehicle_detector
from config.collect_data.directkeys import PressKey,ReleaseKey, W, A, S, D
from config.collect_data.getkeys import key_check
from config.test_model.motion import motion_detection
from keras.applications.inception_v3 import InceptionV3 as googlenet

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

from keras.models import load_model

GAME_WIDTH = 800
GAME_HEIGHT = 600

log_len = 1
motion_req = 400
motion_log = deque(maxlen=log_len)

WIDTH = 400
HEIGHT = 300

MODEL_NAME = 'model_3_400x300_raw_custom'
MODEL_NAME_OBJECT_DETECTOR = "ssd_mobilenet_v2_320x320_coco17_tpu-8"

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
        PressKey(A)
        ReleaseKey(S)
        ReleaseKey(D)
        ReleaseKey(S)

def right():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
        PressKey(D)
        ReleaseKey(A)
        ReleaseKey(S)

def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():

    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(D)


# %%
# Vehicle Detector
# ~~~~~~~~~~~~~~~~~~~~~~~~~
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

from config.collect_data.grabscreen import grab_screen

from config.collect_data.object_detector import download_object_detector
from config.collect_data.object_detector import detect_fn

PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS = download_object_detector(MODEL_NAME_OBJECT_DETECTOR)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


load_name = 'models/{}'.format(MODEL_NAME)
model = tf.keras.models.load_model(load_name)
model.summary()

print('We have loaded a previous model!!!!')

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    mode_choice = 0

    screen = grab_screen(region=(0,40,GAME_WIDTH,GAME_HEIGHT+40))
    screen_processed = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    prev = cv2.resize(screen_processed, (WIDTH,HEIGHT))

    t_minus = prev
    t_now = prev
    t_plus = prev

    while(True):

        if not paused:
            # resize to something a bit more acceptable for a CNN
            screen_processed = cv2.resize(screen, (WIDTH,HEIGHT))
            # run a color convert:
            screen_processed = cv2.cvtColor(screen_processed, cv2.COLOR_BGR2RGB)

            last_time = time.time()

            delta_count_last = motion_detection(t_minus, t_now, t_plus, screen_processed)

            t_minus = t_now
            t_now = t_plus
            t_plus = screen_processed
            t_plus = cv2.blur(t_plus,(4,4))

            screen_processed = screen_processed / 255.0

            prediction = model.predict([screen_processed.reshape(-1,HEIGHT,WIDTH,3)])

            prediction = np.array(prediction) * np.array([0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'
            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

            motion_log.append(delta_count_last)
            motion_avg = round(mean(motion_log),3)
            print('loop took {} seconds. Motion: {}. Choice: {}'.format(round(time.time()-last_time, 3) , motion_avg, choice_picked))

            # If collision warning detected, start evasive maneuvers:
            screen = grab_screen(region=(40,100,800,450))
            image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            last_time = time.time()
            detections, predictions_dict, shapes = detect_fn(detection_model, input_tensor)
            print('Time to load model: {}'.format(time.time() - last_time))

            # label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # Define collision & distance default values
            collision = False
            distance = 1.0
            for i,b in enumerate(detections['detection_boxes'][0]):
                #                                          car                                           bus                                          truck
                if detections['detection_classes'][0][i] == 2 or detections['detection_classes'][0][i] == 5 or detections['detection_classes'][0][i] == 7:
                    if detections['detection_scores'][0][i] >= 0.5:
                        mid_x = (detections['detection_boxes'][0][i][1]+detections['detection_boxes'][0][i][3])/2
                        mid_y = (detections['detection_boxes'][0][i][0]+detections['detection_boxes'][0][i][2])/2
                        aspect_ratio = (detections['detection_boxes'][0][i][3]-detections['detection_boxes'][0][i][1]) / (detections['detection_boxes'][0][i][2]-detections['detection_boxes'][0][i][0])
                        distance = np.round(((1 - (detections['detection_boxes'][0][i][3] - detections['detection_boxes'][0][i][1]))**4),2)

                        # Possible collision
                        if distance <= 0.5 and mid_x > 0.3 and mid_x < 0.7 and aspect_ratio < 2:
                            collision = True
            print('Collision: {} - Distance: {}\n'.format(collision, distance))
            if collision:
                print('Collision Warning. Starting evasive maneuvers: ', end='')
                reverse()
                time.sleep(random.uniform(0,2))

            else:
                pass


            # If vehicle is stucked, start some evasive maneuvers
            if motion_avg < motion_req and len(motion_log) >= log_len:
                print('WERE PROBABLY STUCK FFS, initiating some evasive maneuvers.')

                # 0 = reverse straight, turn left out
                # 1 = reverse straight, turn right out
                # 2 = reverse left, turn right out
                # 3 = reverse right, turn left out

                quick_choice = random.randrange(0,4)

                if quick_choice == 0:
                    reverse()
                    time.sleep(random.uniform(1,2))
                    forward_left()
                    time.sleep(random.uniform(1,2))

                elif quick_choice == 1:
                    reverse()
                    time.sleep(random.uniform(1,2))
                    forward_right()
                    time.sleep(random.uniform(1,2))

                elif quick_choice == 2:
                    reverse_left()
                    time.sleep(random.uniform(1,2))
                    forward_right()
                    time.sleep(random.uniform(1,2))

                elif quick_choice == 3:
                    reverse_right()
                    time.sleep(random.uniform(1,2))
                    forward_left()
                    time.sleep(random.uniform(1,2))

                for i in range(log_len-2):
                    del motion_log[0]

        keys = key_check()

        # Press 'T' to pause execution of script
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                print('\n\nPAUSED\n\n')
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()
