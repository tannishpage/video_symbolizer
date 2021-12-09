import sys
if len(sys.argv) != 3:
    print("Usage: python3 video_symbolizer <path to video files> <output folder path>")
    exit()
ROOT_DIR = "/home/tannishpage/Nextcloud/University 2021/Summer research/All Code/Mask_RCNN-tensorflow2.0"
SORT_DIR = "/home/tannishpage/Documents/Sign_Language_Detection/sort"
sys.path.append(ROOT_DIR) # Adding MRCNN root dir to path to import models
sys.path.append(SORT_DIR) # Adding sort to path to import it

from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgba2rgb
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from mrcnn import config # Importing RCNN
from mrcnn import utils
import mrcnn.model as modellib
import sort # Importing SORT

class InferenceConfig(config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'coco'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

def get_average(landmarks):
    """
    Function calculates the average position of a list of landmarks

    Parameters
        - landmarks (list): List of landmarks
    Return
        A tuple with the average x and y coordinates for a given list of
        landmarks
    """
    avg_x = 0
    avg_y = 0
    for landmark in landmarks:
        avg_x += landmark.x
        avg_y += landmark.y
    return avg_x/len(landmarks), avg_y/len(landmarks)

# Initializing some global variables
VIDEO_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

def main():
    # Read videos
    video_paths = [os.path.join(VIDEO_FOLDER, file)
                    for file in os.listdir(VIDEO_FOLDER)
                    if os.path.isfile(os.path.join(VIDEO_FOLDER, file))]

    output_files = dict()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    # Getting Enums for the landmarks we are interested in
    hand_landmarks_left = [mp_pose.PoseLandmark.LEFT_PINKY,
                           mp_pose.PoseLandmark.LEFT_INDEX,
                           mp_pose.PoseLandmark.LEFT_THUMB]

    hand_landmarks_right = [mp_pose.PoseLandmark.RIGHT_PINKY,
                            mp_pose.PoseLandmark.RIGHT_INDEX,
                            mp_pose.PoseLandmark.RIGHT_THUMB]

    mouth_landmarks = [mp_pose.PoseLandmark.MOUTH_LEFT,
                       mp_pose.PoseLandmark.MOUTH_RIGHT]

    eye_landmarks = [mp_pose.PoseLandmark.LEFT_EYE,
                     mp_pose.PoseLandmark.RIGHT_EYE]

    shoulder_landmarks = [mp_pose.PoseLandmark.LEFT_SHOULDER,
                          mp_pose.PoseLandmark.RIGHT_SHOULDER]

    hip_landmarks = [mp_pose.PoseLandmark.LEFT_HIP,
                     mp_pose.PoseLandmark.RIGHT_HIP]

    conf = InferenceConfig()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=conf)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    for video_path in video_paths:
        video = cv2.VideoCapture(video_path)
        tracker = sort.Sort() #Using default tracker settings
        while True:
            ret, frame = video.read()
            if ret:
                results = model.detect([frame])
                r = results[0]
                indices = np.where(r['class_ids'] == 1) # Getting humans only
                all_humans_bbox = []
                for i, index in enumerate(indices[0]):
                    y1, x1, y2, x2 = r['rois'][index]
                    bbox = (x1, y1, x2, y2, r['scores'][index])
                    all_humans_bbox.append(bbox)
                output = tracker.update(np.array(all_humans_bbox))
                for human in output:
                    x1 = int(human[0])
                    y1 = int(human[1])
                    x2 = int(human[2])
                    y2 = int(human[3])
                    # TODO:
                    # Use mediapipe to get output on person
                    # Then use the rules to get symbol
                    # Store symbol in dict
                    # If the person is no longer being tracked then add a null
                    # to that untracked person

if __name__ == "__main__":
    main()
