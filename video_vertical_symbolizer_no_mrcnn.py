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
    if OUTPUT:
        print("Starting Symbolization Process")
    for i, video_path in enumerate(video_paths):
        if OUTPUT:
            print(f"\nLoading in Video {video_path} ({i} of {num_vids})")
        video_name = video_path.replace("_Filtered.mp4", "") if video_path.endswith("_Filtered.mp4") else video_path.replace(".mkv", "")
        video = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video_path))
        # Dictionary to store symbols for each person
        human_tracked_symbols = dict()
        human_tracked_image = dict() # An image of the person being tracked
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

        key = 0
        while frame_count < total_frames:
            ret, frame = video.read()
            frame_count += 1
            if OUTPUT:
                percentage = frame_count / total_frames * 100
            if ret:
                height, width, _ = frame.shape
                try:
                    mp_pose_results = pose.process(frame)
                    if (mp_pose_results.pose_landmarks == None):
                        if key not in human_tracked_symbols.keys():
                            # Storing left and right symbols
                            human_tracked_symbols[key] = (["H"],
                                                          ["H"],
                                                          [str(frame_count)],
                                                          [groundtruth[video_name, frame_count]])
                        else:
                            human_tracked_symbols[key][0].append("H")
                            human_tracked_symbols[key][1].append("H")
                            human_tracked_symbols[key][2].append(str(frame_count))
                            human_tracked_symbols[key][3].append(groundtruth[video_name, frame_count])

                        if key not in human_tracked_image.keys():
                            human_tracked_image[key] = frame

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
                    symbols["E"] = (two_third_shoulder_hip[1],
                                    third_shoulder_hip[1])
                    symbols["F"] = (hip[1], third_shoulder_hip[1])
                    symbols["G"] = (height, hip[1])
                    #Check which region hands are in
                    left_flag = False
                    right_flag = False
                    left_symbol = ""
                    right_symbol = ""
                    for symbol in symbols.keys():
                        if check(symbols[symbol],
                                 left_hand[1]) and not left_flag:
                            left_symbol = symbol
                            left_flag = True
                            if VIS:
                                person_frame = cv2.circle(person_frame,
                                                    (int(left_hand[0]*width),
                                                      int(left_hand[1]*height)),
                                                    2,
                                                    colors[symbol],
                                                    2)

                        if check(symbols[symbol],
                                 right_hand[1]) and not right_flag:
                            right_symbol = symbol
                            right_flag = True
                            if VIS:
                                person_frame = cv2.circle(person_frame,
                                                    (int(right_hand[0]*width),
                                                     int(right_hand[1]*height)),
                                                    2,
                                                    colors[symbol],
                                                    2)

                        if left_flag and right_flag:
                            break
                    if key not in human_tracked_symbols.keys():
                        # Storing left and right symbols
                        human_tracked_symbols[key] = ([left_symbol],
                                                      [right_symbol],
                                                      [str(frame_count)],
                                                      [groundtruth[video_name, frame_count]])
                    else:
                        human_tracked_symbols[key][0].append(left_symbol)
                        human_tracked_symbols[key][1].append(right_symbol)
                        human_tracked_symbols[key][2].append(str(frame_count))
                        human_tracked_symbols[key][3].append(groundtruth[video_name, frame_count])

                    if key not in human_tracked_image.keys():
                        human_tracked_image[key] = frame

                    if VIS:
                        person_frame = cv2.line(person_frame,
                                        (0,
                                         int(third_shoulder_hip[1]*height)),
                                        (int(width),
                                         int(third_shoulder_hip[1]*height)),
                                        (0, 255, 0),
                                        thickness=2)
                        person_frame = cv2.line(person_frame,
                                        (0,
                                         int(two_third_shoulder_hip[1]*height)),
                                        (int(width),
                                         int(two_third_shoulder_hip[1]*height)),
                                        (0, 255, 0),
                                        thickness=2)
                        person_frame = cv2.line(person_frame,
                                        (0,
                                         int(shoulder[1]*height)),
                                        (int(width),
                                         int(shoulder[1]*height)),
                                        (0, 255, 0),
                                        thickness=2)
                        person_frame = cv2.line(person_frame,
                                        (0,
                                         int(hip[1]*height)),
                                        (int(width),
                                         int(hip[1]*height)),
                                        (0, 255, 0),
                                        thickness=2)
                        person_frame = cv2.line(person_frame,
                                        (0,
                                         int(eyes[1]*height)),
                                        (int(width),
                                         int(eyes[1]*height)),
                                        (0, 255, 0),
                                        thickness=2)
                        person_frame = cv2.line(person_frame,
                                        (0,
                                         int(mouth[1]*height)),
                                        (int(width),
                                         int(mouth[1]*height)),
                                        (0, 255, 0),
                                        thickness=2)
                        cv2.imshow(str(key), person_frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    if OUTPUT:
                        sys.stdout.write("\r[{}{}] {:.2f}% {}/{}".format('='*int(percentage/2),
                                                        '.' *(50 - int(percentage/2)),
                                                        percentage, frame_count,
                                                        total_frames))
                    sys.stdout.flush()
                except Exception as e:
                    print(datetime.datetime.now(), e, video_path, frame_count, frame, frame.shape)
            else:
                break

        for key in human_tracked_image.keys():
            symbol_file = open(os.path.join(OUTPUT_FOLDER, video_path.split('.')[0], f"{video_path.split('.')[0]}.{int(key)}.txt"), 'w')
            symbol_file.write(f"frame:{','.join(human_tracked_symbols[key][2])}\nleft:{','.join(human_tracked_symbols[key][0])}\nright:{','.join(human_tracked_symbols[key][1])}\nlabel:{','.join(human_tracked_symbols[key][3])}")
            symbol_file.close()
            if human_tracked_image[key].size != 0:
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, video_path.split('.')[0], f"{video_path.split('.')[0]}.{int(key)}.jpg"), human_tracked_image[key])

if __name__ == "__main__":
    main()
