"""
    Takes too long to run on a single video. Plus barely extracts any features.
    Pick specific videos that have good frontal views, and extract symbols from
    ~ 5000 frames per video.
"""

import sys
if len(sys.argv) < 3:
    print("Usage: python3 video_symbolizer \
<path to video files> <output folder path> [--vis --output]")
    exit()
ROOT_DIR = ""
SORT_DIR = ""
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

def get_landmark_values(landmark_enums, results):
    """
    Returns x, y coordinates of the landmark values corresponding to the
    landmark_enums

    Parameters
        - landmark_enums (list): A list of enums for the landmarks
        - results : the return value from mediapipe's mp_pose.Pose.process()
    Returns
        The x, y coordinates of the average position of the landmarks passed
    """
    values = []
    for enum in landmark_enums:
        values.append(results.pose_landmarks.landmark[enum])
    return get_average(values)

# Initializing some global variables
VIDEO_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
VIS = "--vis" in sys.argv
OUTPUT = "--output" in sys.argv
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

def main():
    # Read videos
    video_paths = [file for file in os.listdir(VIDEO_FOLDER)
                    if file.endswith(".mp4") or file.endswith(".mkv")]#os.path.isfile(os.path.join(VIDEO_FOLDER, file))]

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
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=MODEL_DIR,
                              config=conf)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    symbols = {"A":(), "B":(), "C":(), "D":(), "E":(), "F":(), "G":()}
    if VIS:
        colors = {"A":(225, 0, 0), "B":(0, 225, 0), "C":(0, 0, 225), "D":(225, 225, 0), "E":(0, 225, 225), "F":(225, 0, 225), "G":(225, 225, 225)}
    check = lambda limits, pos: (pos < limits[0]) and (pos > limits[1])
    if OUTPUT:
        print("Starting Symbolization Process")
    for video_path in video_paths:
        if OUTPUT:
            print(f"\nLoading in Video {video_path}")
        video = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_path))
        tracker = sort.Sort() #Using default tracker settings
        # Dictionary to store symbols for each person
        human_tracked_symbols = dict()
        human_tracked_image = dict() # An image of the person being tracked
        if OUTPUT:
            frame_count = 0
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            percentage = frame_count / total_frames * 100
            sys.stdout.write("\r[{}{}] {:.2f}% {}/{}".format('='*int(percentage/2),
                                                '.' *(50 - int(percentage/2)),
                                                percentage, frame_count,
                                                total_frames))
            sys.stdout.flush()
        while frame_count < 300:
            ret, frame = video.read()
            if OUTPUT:
                frame_count += 1
                percentage = frame_count / total_frames * 100
            if ret:
                frame_height, frame_width, _ = frame.shape
                RCNN_results = model.detect([frame])
                r = RCNN_results[0]
                indices = np.where(r['scores'] >= 0.90)
                indices = np.where(r['class_ids'][indices[0]] == 1) # Getting human class only
                all_humans_bbox = []
                for i, index in enumerate(indices[0]):
                    y1, x1, y2, x2 = r['rois'][index]
                    bbox = (x1, y1, x2, y2, r['scores'][index])
                    all_humans_bbox.append(bbox)
                    if VIS:
                        pass#cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                if VIS:
                    pass#cv2.imshow("Window", frame)

                if all_humans_bbox == []:
                    output = tracker.update()
                else:
                    output = tracker.update(np.array(all_humans_bbox))
                for human in output:
                    height = y2-y1
                    width = x2-x1
                    x1 = int(human[0])
                    y1 = int(human[1])
                    x2 = int(human[2])
                    y2 = int(human[3])
                    key = human[4]
                    if key != 1:
                        continue
                    if (x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0):
                        if key not in human_tracked_symbols.keys():
                            # Storing left and right symbols
                            human_tracked_symbols[key] = (["H"],
                                                          ["H"])
                        else:
                            human_tracked_symbols[key][0].append("H")
                            human_tracked_symbols[key][1].append("H")

                        if key not in human_tracked_image.keys():
                            human_tracked_image[key] = frame[y1:y2, x1:x2]
                        continue
                    person_frame = frame[y1:y2, x1:x2]
                    height, width, _ = person_frame.shape
                    mp_pose_results = pose.process(person_frame)
                    if (mp_pose_results.pose_landmarks == None):
                        if key not in human_tracked_symbols.keys():
                            # Storing left and right symbols
                            human_tracked_symbols[key] = (["H"],
                                                          ["H"])
                        else:
                            human_tracked_symbols[key][0].append("H")
                            human_tracked_symbols[key][1].append("H")

                        if key not in human_tracked_image.keys():
                            human_tracked_image[key] = frame[y1:y2, x1:x2]
                        continue

                    left_hand = get_landmark_values(hand_landmarks_left,
                                                    mp_pose_results)
                    right_hand = get_landmark_values(hand_landmarks_right,
                                                     mp_pose_results)
                    mouth = get_landmark_values(mouth_landmarks,
                                                mp_pose_results)
                    eyes = get_landmark_values(eye_landmarks,
                                               mp_pose_results)
                    shoulder = get_landmark_values(shoulder_landmarks,
                                                   mp_pose_results)
                    hip = get_landmark_values(hip_landmarks,
                                              mp_pose_results)
                    shoulder_hip_distance = (hip[0] - shoulder[0],
                                             hip[1] - shoulder[1])

                    third_shoulder_hip = (shoulder[0] +\
                                            shoulder_hip_distance[0]/3,
                                          shoulder[1] +\
                                            shoulder_hip_distance[1]/3)

                    two_third_shoulder_hip = (shoulder[0] +\
                                                2*shoulder_hip_distance[0]/3,
                                              shoulder[1] +\
                                                2*shoulder_hip_distance[1]/3)
                    symbols["A"] = (eyes[1], 0)
                    symbols["B"] = (mouth[1], eyes[1])
                    symbols["C"] = (shoulder[1], mouth[1])
                    symbols["D"] = (third_shoulder_hip[1], shoulder[1])
                    symbols["E"] = (two_third_shoulder_hip[1], third_shoulder_hip[1])
                    symbols["F"] = (hip[1], third_shoulder_hip[1])
                    symbols["G"] = (height, hip[1])
                    #Check which region hands are in
                    left_flag = False
                    right_flag = False
                    left_symbol = ""
                    right_symbol = ""
                    for symbol in symbols.keys():
                        if check(symbols[symbol], left_hand[1]) and not left_flag:
                            left_symbol = symbol
                            left_flag = True
                            if VIS:
                                person_frame = cv2.circle(person_frame,(int(left_hand[0]*width),int(left_hand[1]*height)) , 2, colors[symbol], 2)

                        if check(symbols[symbol], right_hand[1]) and not right_flag:
                            right_symbol = symbol
                            right_flag = True
                            if VIS:
                                person_frame = cv2.circle(person_frame,(int(right_hand[0]*width),int(right_hand[1]*height)) , 2, colors[symbol], 2)

                        if left_flag and right_flag:
                            break
                    if key not in human_tracked_symbols.keys():
                        # Storing left and right symbols
                        human_tracked_symbols[key] = ([left_symbol],
                                                      [right_symbol])
                    else:
                        human_tracked_symbols[key][0].append(left_symbol)
                        human_tracked_symbols[key][1].append(right_symbol)

                    if key not in human_tracked_image.keys():
                        human_tracked_image[key] = frame[y1:y2, x1:x2]

                    if VIS:
                        person_frame = cv2.line(person_frame, (0, int(third_shoulder_hip[1]*height)), (int(width), int(third_shoulder_hip[1]*height)) ,(0, 255, 0), thickness=2)
                        person_frame = cv2.line(person_frame, (0, int(two_third_shoulder_hip[1]*height)), (int(width), int(two_third_shoulder_hip[1]*height)) ,(0, 255, 0), thickness=2)
                        person_frame = cv2.line(person_frame, (0, int(shoulder[1]*height)), (int(width), int(shoulder[1]*height)) , (0, 255, 0), thickness=2)
                        person_frame = cv2.line(person_frame, (0, int(hip[1]*height)), (int(width), int(hip[1]*height)) , (0, 255, 0), thickness=2)
                        person_frame = cv2.line(person_frame, (0, int(eyes[1]*height)), (int(width), int(eyes[1]*height)) ,(0, 255, 0), thickness=2)
                        person_frame = cv2.line(person_frame, (0, int(mouth[1]*height)), (int(width), int(mouth[1]*height)) , (0, 255, 0), thickness=2)
                    if VIS:
                        cv2.imshow(str(key), person_frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                if OUTPUT:
                    sys.stdout.write("\r[{}{}] {:.2f}% {}/{}".format('='*int(percentage/2),
                                                        '.' *(50 - int(percentage/2)),
                                                        percentage, frame_count,
                                                        total_frames))
                    sys.stdout.flush()
            else:
                break
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        if not os.path.exists(os.path.join(OUTPUT_FOLDER, video_path)):
            os.mkdir(os.path.join(OUTPUT_FOLDER, video_path.split('.')[0]))
        for key in human_tracked_image.keys():
            symbol_file = open(os.path.join(OUTPUT_FOLDER, video_path.split('.')[0], f"{video_path.split('.')[0]}.{int(key)}.txt"), 'w')
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, video_path.split('.')[0], f"{video_path.split('.')[0]}.{int(key)}.jpg"), human_tracked_image[key])
            symbol_file.write(f"left:{','.join(human_tracked_symbols[key][0])}\nright:{','.join(human_tracked_symbols[key][1])}\n")
            symbol_file.close()


if __name__ == "__main__":
    main()
