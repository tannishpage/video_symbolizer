import sys
import os
import sys
import random
import math
import numpy as np
import cv2
import mediapipe as mp
import datetime
import pytictoc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

def check_cmd_arguments(arg, default, false_value):
    arg_value = false_value # Setting the argument's value to the false value
    # Checking if argument is in the sys args
    if arg in sys.argv:
        index = sys.argv.index(arg) + 1 # Grab the index of the value for arg
        if index >= len(sys.argv):
        # If the value isn't passed, set it to the default value
            arg_value = default
        else:
            # We check that the value isn't another argument
            value = sys.argv[index]
            if "-" not in value:
                arg_value = value # Assign the value
            else:
                arg_value = default # else we use use the default value

    return arg_value

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


def convert_csv_to_list_of_tuple(datax, datay, height=1, width=1):
    data = []
    for x, y, in zip(datax, datay):
        data.append(tuple((int(x*width), int(y*height))))
    return data

def find_points(videos, csv_save_loc):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    # Getting Enums for the hands for tracking
    hand_landmarks_left = [mp_pose.PoseLandmark.LEFT_PINKY,
                           mp_pose.PoseLandmark.LEFT_INDEX,
                           mp_pose.PoseLandmark.LEFT_THUMB]

    hand_landmarks_right = [mp_pose.PoseLandmark.RIGHT_PINKY,
                            mp_pose.PoseLandmark.RIGHT_INDEX,
                            mp_pose.PoseLandmark.RIGHT_THUMB]

    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    for video_path in videos:
        video = cv2.VideoCapture(video_path)
        video_loc = os.path.basename(video_path)
        video_name = video_loc.replace(".mp4", "") if video_loc.endswith(".mp4") else video_loc.replace(".mkv", "")
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('./outputvid2.avi',fourcc, 25.0, (1920,1080))
        frame_count = 0
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        data = {}
        data["Left Hand.x"] = []
        data["Right Hand.x"] = []
        data["Left Hand.y"] = []
        data["Right Hand.y"] = []

        while frame_count < total_frames:
            ret, frame = video.read()
            frame_count += 1
            if ret:
                sys.stdout.write(f"\rProcessing {video_loc}: {(frame_count/total_frames)*100:.2f}%")
                height, width, _ = frame.shape
                mp_pose_results = pose.process(frame)
                if (mp_pose_results.pose_landmarks == None):
                    continue
                left_hand = get_landmark_values(hand_landmarks_left, mp_pose_results)
                right_hand = get_landmark_values(hand_landmarks_right, mp_pose_results)
                # Leave the hand positions in their normalized form
                # It will make it easier to compare with other videos
                data["Left Hand.x"].append(left_hand[0])
                data["Left Hand.y"].append(left_hand[1])
                data["Right Hand.x"].append(right_hand[0])
                data["Right Hand.y"].append(right_hand[1])

        else:
            data_frame = pd.DataFrame(data)
            data_frame.to_csv(os.path.join(csv_save_loc, f"{video_name}.csv"))
            print()

def main():
    USAGE ="""Usage: python3 signer_heatmap.py <path_to_video_file> [options]
   OR: python3 signer_heatmap.py --from_csv <path_to_csv_file> [options]
        -h, --help          Display this help message
        --save_csv          The location to save the csv file
        --radius            The radius to determin popularith (in pixels)
        --image             The image to draw heatmap on
"""
    csv_save_loc = check_cmd_arguments("--save_csv", "./", "./")
    background_image_loc = check_cmd_arguments("--image", False, False)
    find_points([sys.argv[1]], csv_save_loc)

if __name__ == "__main__":
    main()
