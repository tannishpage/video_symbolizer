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


def main():
    video_loc = "/home/tannishpage/Videos/test.mp4"

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

    video = cv2.VideoCapture(video_loc)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('./outputvid2.avi',fourcc, 25.0, (1920,1080))
    frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    data = {} # TODO make dict and convert to pandas dataframe and save as csv
    data["Left Hand.x"] = []
    data["Right Hand.x"] = []
    data["Left Hand.y"] = []
    data["Right Hand.y"] = []

    while frame_count < total_frames:
        ret, frame = video.read()
        frame_count += 1
        if ret:
            sys.stdout.write(f"\r{frame_count/total_frames * 100}%")
            height, width, _ = frame.shape
            mp_pose_results = pose.process(frame)
            if (mp_pose_results.pose_landmarks == None):
                continue
            left_hand = get_landmark_values(hand_landmarks_left, mp_pose_results)
            right_hand = get_landmark_values(hand_landmarks_right, mp_pose_results)
            # Leave the hand positions in their normalized form
            # It will make it easier to compare with other videos
            # TODO: Save in CSV format
            data["Left Hand.x"].append(left_hand[0])
            data["Left Hand.y"].append(left_hand[1])
            data["Right Hand.x"].append(right_hand[0])
            data["Right Hand.y"].append(right_hand[1])

    else:
        data_frame = pd.DataFrame(data)
        data_frame.to_csv("./test.csv")
        print()

if __name__ == "__main__":
    main()
