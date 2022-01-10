import sys
if len(sys.argv) < 4:
    print("Usage: python3 video_symbolizer \
<path to video files> <output folder path> <groundtruth file> [--vis --output]")
    exit()

import os
import sys
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
import datetime

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

def get_groundtruth_data(groundtruth_file):
    gt = {}
    have_groundtruth_data = False
    if len(groundtruth_file) > 0:
        try:
            # open and load the groundtruth data
            print('Loading groundtruth data...')
            with open(groundtruth_file, 'r') as gt_file:
                gt_lines = gt_file.readlines()
            for gtl in gt_lines:
                gtf = gtl.rstrip().split(' ')
                #gt file has 3 items per line (vid_ID, frame ID, label)
                if len(gtf) == 3:
                    gt[(gtf[0], int(gtf[1]))] = gtf[2]
            print('ok\n')
            have_groundtruth_data = True
        except:
            pass
    return gt

# Initializing some global variables
VIDEO_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2]
VIS = "--vis" in sys.argv
OUTPUT = "--output" in sys.argv

def main():
    # Read videos
    video_paths = [file for file in os.listdir(VIDEO_FOLDER)
                    if file.endswith("_Filtered.mp4")]
    num_vids = len(video_paths)

    groundtruth = get_groundtruth_data(sys.argv[3])

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

    # Initializing mediapipe pose
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    symbols = {"A":(), "B":(), "C":(), "D":(), "E":(), "F":(), "G":()}

    if VIS:
        colors = {"A":(225, 0, 0),
                  "B":(0, 225, 0),
                  "C":(0, 0, 225),
                  "D":(225, 225, 0),
                  "E":(0, 225, 225),
                  "F":(225, 0, 225),
                  "G":(225, 225, 225)}

    check = lambda limits, pos: (pos < limits[0]) and (pos > limits[1])
    on_left = lambda center, pos: pos < center

    if OUTPUT:
        print("Starting Symbolization Process")
    for i, video_path in enumerate(video_paths):
        if OUTPUT:
            print(f"\nLoading in Video {video_path} ({i} of {num_vids})")
        video_name = video_path.replace("_Filtered.mp4", "") if video_path.endswith("_Filtered.mp4") else video_path.replace(".mkv", "")
        video = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_path))
        # List to store symbols and other information
        # Left Symbol, Right Symbol, Frame Count, Ground Truth Value
        human_tracked_symbols = [[], [], [], []]
        frame_count = 0
        if OUTPUT:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            percentage = frame_count / total_frames * 100
            sys.stdout.write("\r[{}{}] {:.2f}% {}/{}".format(
                                                '='*int(percentage/2),
                                                '.' *(50 - int(percentage/2)),
                                                percentage, frame_count,
                                                total_frames))
            sys.stdout.flush()

        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        if not os.path.exists(os.path.join(OUTPUT_FOLDER, video_path.split('.')[0])):
            os.mkdir(os.path.join(OUTPUT_FOLDER, video_path.split('.')[0]))
        else:
            print(f"\nSkipping video {video_path}")
            continue
