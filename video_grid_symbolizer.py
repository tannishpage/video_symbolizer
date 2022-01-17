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
import cv2
import mediapipe as mp
import datetime
import pytictoc

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

def get_symbol(symbols, hand):
    for symbol in symbols.keys():
        if check(symbols[symbol], hand[1]):
            return symbol


# Lambda functions because they are simple checks
check = lambda limits, pos: (pos < limits[0]) and (pos > limits[1])
on_left = lambda center, pos: pos < center
def main():
    time = pytictoc.TicToc()
    VIDEO_FOLDER = sys.argv[1]
    OUTPUT_FOLDER = sys.argv[2]
    VIS = "--vis" in sys.argv
    OUTPUT = "--output" in sys.argv
    # Read videos
    if os.path.isdir(VIDEO_FOLDER):
        video_paths = [file for file in os.listdir(VIDEO_FOLDER)
                    if file.endswith(".mp4") or file.endswith(".mkv")]
    else:
        # If not direcotry, then assume it's a single file.
        # Allows user to pass single file to be symbolized
        video_paths = [os.path.basename(VIDEO_FOLDER)]
        VIDEO_FOLDER = os.path.dirname(VIDEO_FOLDER)
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

    shoulder_landmarks = [mp_pose.PoseLandmark.LEFT_SHOULDER,
                          mp_pose.PoseLandmark.RIGHT_SHOULDER]

    hip_landmarks = [mp_pose.PoseLandmark.LEFT_HIP,
                     mp_pose.PoseLandmark.RIGHT_HIP]

    # Initializing mediapipe pose
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    # Since we have 2 sides, I've split up the symbols dict into 2 dicts
    # It's easier to deal with
    symbols_left = {"A":(), "C":(), "E":(), "G":()}
    symbols_right = {"B":(), "D":(), "F":(), "G":()}

    if VIS:
        colors = {"A":(225, 0, 0),
                  "B":(0, 225, 0),
                  "C":(0, 0, 225),
                  "D":(225, 225, 0),
                  "E":(0, 225, 225),
                  "F":(225, 0, 225),
                  "G":(225, 225, 225)}

    if OUTPUT:
        print("Starting Symbolization Process")
    for i, video_path in enumerate(video_paths):
        if OUTPUT:
            print(f"\nLoading in Video {video_path} ({i} of {num_vids})")
        video_name = video_path.replace(".mp4", "") if video_path.endswith(".mp4") else video_path.replace(".mkv", "")
        video = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_path))
        # List to store symbols and other information
        # Left Symbol, Right Symbol, Frame Count, Ground Truth Value
        human_tracked_symbols = [[], [], [], []]
        frame_count = 0
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if OUTPUT:
            percentage = frame_count / total_frames * 100
            sys.stdout.write("\r[{}{}] {:.2f}% {}/{}".format(
                                                '='*int(percentage/2),
                                                '.' *(50 - int(percentage/2)),
                                                percentage, frame_count,
                                                total_frames))
            sys.stdout.flush()
            time.tic()

        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        while frame_count < total_frames:
            ret, frame = video.read()
            frame_count += 1
            if OUTPUT:
                percentage = frame_count / total_frames * 100
            if ret:
                height, width, _ = frame.shape
                mp_pose_results = pose.process(frame)
                if (mp_pose_results.pose_landmarks == None):
                    human_tracked_symbols[0].append("H")
                    human_tracked_symbols[1].append("H")
                    human_tracked_symbols[2].append(str(frame_count))
                    human_tracked_symbols[3].append(groundtruth.get((video_name, frame_count), -1))
                    continue

                left_hand = get_landmark_values(hand_landmarks_left,
                                                mp_pose_results)
                right_hand = get_landmark_values(hand_landmarks_right,
                                                 mp_pose_results)
                shoulder = get_landmark_values(shoulder_landmarks,
                                               mp_pose_results)
                hip = get_landmark_values(hip_landmarks,
                                          mp_pose_results)
                shoulder_hip_distance = (hip[0] - shoulder[0],
                                         hip[1] - shoulder[1])

                half_shoulder_hip = (shoulder[0] +\
                                        shoulder_hip_distance[0]/2,
                                     shoulder[1] +\
                                        shoulder_hip_distance[1]/2)

                # Setting boundries for both the left and right side
                symbols_left["A"] = symbols_right["B"] = (shoulder[1], 0)
                symbols_left["C"] = symbols_right["D"] = (half_shoulder_hip[1], shoulder[1])
                symbols_left["E"] = symbols_right["F"] = (hip[1], half_shoulder_hip[1])
                symbols_left["G"] = symbols_right["G"] = (height, hip[1])

                # TODO: Find which side each hand is in, and assign the
                #       relevant symbol

                if on_left(left_hand[0], shoulder[0]):
                    left_symbol = get_symbol(symbols_left, left_hand)
                else:
                    left_symbol = get_symbol(symbols_right, right_hand)

                if on_left(right_hand[0], shoulder[0]):
                    right_symbol = get_symbol(symbols_left, right_hand)
                else:
                    right_symbol = get_symbol(symbols_right, right_hand)

                human_tracked_symbols[0].append(left_symbol)
                human_tracked_symbols[1].append(right_symbol)
                human_tracked_symbols[2].append(str(frame_count))
                human_tracked_symbols[3].append(groundtruth.get((video_name, frame_count), -1))

                if VIS:
                    frame = cv2.line(frame,
                                    (0,
                                     int(half_shoulder_hip[1]*height)),
                                    (int(width),
                                     int(half_shoulder_hip[1]*height)),
                                    (0, 255, 0),
                                    thickness=2)
                    frame = cv2.line(frame,
                                    (0,
                                     int(shoulder[1]*height)),
                                    (int(width),
                                     int(shoulder[1]*height)),
                                    (0, 255, 0),
                                    thickness=2)
                    frame = cv2.line(frame,
                                    (0,
                                     int(hip[1]*height)),
                                    (int(width),
                                     int(hip[1]*height)),
                                    (0, 255, 0),
                                    thickness=2)
                    frame = cv2.line(frame,
                                    (int(shoulder[0]*width),
                                     0),
                                    (int(shoulder[0]*width),
                                     int(hip[1]*height)),
                                    (0, 255, 0),
                                    thickness=2)
                    frame = cv2.circle(frame,
                                      (int(left_hand[0]*width),
                                       int(left_hand[1]*height)),
                                      2,
                                      colors[left_symbol],
                                      2)

                    frame = cv2.circle(frame,
                                      (int(right_hand[0]*width),
                                       int(right_hand[1]*height)),
                                      2,
                                      colors[right_symbol],
                                      2)
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                if OUTPUT:
                    sys.stdout.write("\r[{}{}] {:.2f}% {}/{}".format('='*int(percentage/2),
                                                    '.' *(50 - int(percentage/2)),
                                                    percentage, frame_count,
                                                    total_frames))
                sys.stdout.flush()

        if OUTPUT:
            time.toc()
            print(f"Saving symbolized data to {video_path.split('.')[0]}.txt...")
        symbol_file = open(os.path.join(OUTPUT_FOLDER, f"{video_path.split('.')[0]}.txt"), 'w')
        symbol_file.write(f"frame:{','.join(human_tracked_symbols[2])}\nleft:{','.join(human_tracked_symbols[0])}\nright:{','.join(human_tracked_symbols[1])}\nlabel:{','.join(human_tracked_symbols[3])}")
        symbol_file.close()
        if OUTPUT:
            print("Done")
if __name__ == "__main__":
    main()
