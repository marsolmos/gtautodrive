#!/usr/bin/env python
# coding: utf-8
"""
Detect Objects on Videos
================================
"""


import os
import tarfile
import urllib.request

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

import cv2
import numpy as np
from config.collect_data.grabscreen import grab_screen
import time


def vehicle_detector():
    '''
    Detect vehicles on video stream using TF Object Detection API.
    1- Download object detection pre-trained model from TF Model Zoo
    2- Pass video stream to TF API and get object detection_scores
    3- Visualize boxes with detected objects
    4- Define approximate distance to objects in users trajectory (not on the
       sides)
    5- Return distance to objects in case of possible collision detected
    '''
    # %%
    # Create the data directory
    # ~~~~~~~~~~~~~~~~~~~~~~~~~
    # The snippet shown below will create the ``data`` directory where all our data will be stored. The
    # code will create a directory structure as shown bellow:
    #
    # .. code-block:: bash
    #
    #     data
    #     └── models
    #
    # where the ``models`` folder will will contain the downloaded models.

    DATA_DIR = os.path.join(os.getcwd(), 'config/collect_data/vehicle_detector_data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    for dir in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # %%
    # Download the model
    # ~~~~~~~~~~~~~~~~~~
    # The code snippet shown below is used to download the object detection model checkpoint file,
    # as well as the labels file (.pbtxt) which contains a list of strings used to add the correct
    # label to each detection (e.g. person).
    #
    # The particular detection algorithm we will use is the `SSD MobileNet V2 320x320`. More
    # models can be found in the `TensorFlow 2 Detection Model Zoo <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>`_.
    # To use a different model you will need the URL name of the specific model. This can be done as
    # follows:
    #
    # 1. Right click on the `Model name` of the model you would like to use;
    # 2. Click on `Copy link address` to copy the download link of the model;
    # 3. Paste the link in a text editor of your choice. You should observe a link similar to ``download.tensorflow.org/models/object_detection/tf2/YYYYYYYY/XXXXXXXXX.tar.gz``;
    # 4. Copy the ``XXXXXXXXX`` part of the link and use it to replace the value of the ``MODEL_NAME`` variable in the code shown below;
    # 5. Copy the ``YYYYYYYY`` part of the link and use it to replace the value of the ``MODEL_DATE`` variable in the code shown below.
    #
    # For example, the download link for the model used below is: ``download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz``

    # Download and extract model
    MODEL_DATE = '20200711'
    MODEL_NAME = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
    MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'
    MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
    MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME
    PATH_TO_MODEL_TAR = os.path.join(MODELS_DIR, MODEL_TAR_FILENAME)
    PATH_TO_CKPT = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'checkpoint/'))
    PATH_TO_CFG = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, 'pipeline.config'))
    if not os.path.exists(PATH_TO_CKPT):
        print('Downloading model. This may take a while... ', end='')
        urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
        tar_file = tarfile.open(PATH_TO_MODEL_TAR)
        tar_file.extractall(MODELS_DIR)
        tar_file.close()
        os.remove(PATH_TO_MODEL_TAR)
        print('Done')

    # Download labels file
    LABEL_FILENAME = 'mscoco_label_map.pbtxt'
    LABELS_DOWNLOAD_BASE = \
        'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
    if not os.path.exists(PATH_TO_LABELS):
        print('Downloading label file... ', end='')
        urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
        print('Done')

    # %%
    # Load the model
    # ~~~~~~~~~~~~~~
    # Enable GPU dynamic memory allocation
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    # @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])


    # %%
    # Load label map data (for plotting)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Label maps correspond index numbers to category names, so that when our convolution network
    # predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
    # functions, but anything that returns a dictionary mapping integers to appropriate string labels
    # would be fine.
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    # %%
    # Define the video stream
    # ~~~~~~~~~~~~~~~~~~~~~~~
    # We will use `OpenCV <https://pypi.org/project/opencv-python/>`_ to capture the video stream.
    # For more information you can refer to the `OpenCV-Python Tutorials <https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#capture-video-from-camera>`_

    # %%
    # Putting everything together
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The code shown below loads an image, runs it through the detection model and visualizes the
    # detection results, including the keypoints.
    #
    # Note that this will take a long time (several minutes) the first time you run this code due to
    # tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
    # faster.
    #
    # Here are some simple things to try out if you are curious:
    #
    # * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
    # * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
    # * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.

    # while True:
    # Read frame from screen
    # screen = cv2.resize(grab_screen(region=(40,40,800,450)), (320,320))
    screen = grab_screen(region=(40,100,800,450))
    image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    last_time = time.time()
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    print('Time to load model: {}'.format(time.time() - last_time))

    # label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #       image_np_with_detections,
    #       detections['detection_boxes'][0].numpy(),
    #       (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    #       detections['detection_scores'][0].numpy(),
    #       category_index,
    #       use_normalized_coordinates=True,
    #       max_boxes_to_draw=20,
    #       min_score_thresh=.50,
    #       agnostic_mode=False)

    # Define collision & apx_distance default values
    collision = False
    apx_distance = 1.0
    for i,b in enumerate(detections['detection_boxes'][0]):
        #                                          car                                           bus                                          truck
        if detections['detection_classes'][0][i] == 2 or detections['detection_classes'][0][i] == 5 or detections['detection_classes'][0][i] == 7:
            if detections['detection_scores'][0][i] >= 0.5:
                mid_x = (detections['detection_boxes'][0][i][1]+detections['detection_boxes'][0][i][3])/2
                mid_y = (detections['detection_boxes'][0][i][0]+detections['detection_boxes'][0][i][2])/2
                aspect_ratio = (detections['detection_boxes'][0][i][3]-detections['detection_boxes'][0][i][1]) / (detections['detection_boxes'][0][i][2]-detections['detection_boxes'][0][i][0])
                apx_distance = np.round(((1 - (detections['detection_boxes'][0][i][3] - detections['detection_boxes'][0][i][1]))**4),2)
                # cv2.putText(image_np_with_detections, '{}'.format(str(apx_distance)), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # Possible collision
                if apx_distance <= 0.5 and mid_x > 0.3 and mid_x < 0.7 and aspect_ratio < 2:
                    collision = True
                    return np.array([collision, apx_distance])
                    # cv2.putText(image_np_with_detections, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)


#     # Display output
#     cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800,450)))
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#
# screen.release()
# cv2.destroyAllWindows()

    return np.array([collision, apx_distance])
