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

def distance(point1, point2):
    delta_x = point1[0] - point2[0]
    delta_y = point1[1] - point2[1]

    return np.math.sqrt(delta_x**2 + delta_y**2)

def create_points_in_radius_dictionary(points, radius):
    all_points_in_radius = {}
    count = 0
    total = len(points)
    start = time.time()
    for point in points:
        points_in_radius = []
        count += 1
        sys.stdout.write(f"\rProgress: {count}/{total} {count/total * 100:.2f}% Elapsed Time: {time.time() - start:.2f}s")
        for other_point in points:
            if other_point == point:
                continue
            if distance(point, other_point) < radius:
                points_in_radius.append(other_point)
        all_points_in_radius[point] = points_in_radius
    return all_points_in_radius

def create_heatmap(all_points_in_radius, background_image, total_points, radius):
    height, width, _ = background_image.shape
    for y in range(0, height):
        for x in range(0, width):
            color = background_image[y, x]
            prob = len(all_points_in_radius.get((x, y), []))/total_points
            #color[0] = 255*prob
            color[1] = 0
            color[2] = 255*prob
            background_image[y-1:y+1, x-1:x+1] = color
    return background_image



    """for point in all_points_in_radius:
        for point_in_radius in all_points_in_radius[point]:
            loc = int(point_in_radius[1]*height), int(point_in_radius[0]*width)
            color = background_image[loc]
            prob = len(all_points_in_radius[point])/total_points
            color[2] = 255*prob
            background_image[loc] = color
    return background_image"""



def main():
    USAGE ="""Usage: python3 signer_heatmap.py <path_to_video_file> [options]
   OR: python3 signer_heatmap.py --from_csv <path_to_csv_file> [options]
        -h, --help          Display this help message
        --from_csv          Use this to create a heatmap directly from csv
        --save_csv          The location to save the csv file
        --radius            The radius to determin popularith (in pixels)
        --image             The image to draw heatmap on
"""
    from_csv = check_cmd_arguments("--from_csv", "", False)
    csv_save_loc = check_cmd_arguments("--save_csv", "./", "./")
    background_image_loc = check_cmd_arguments("--image", False, False)
    if from_csv == False:
        find_points([sys.argv[1]], csv_save_loc)
    else:
        if from_csv == "":
            print("--from_csv: Must pass csv file to create heatmap from")
            exit()
        else:
            radius = int(check_cmd_arguments("--radius", 20, 20))
            data = pd.read_csv(from_csv)
            if background_image_loc == False:
                print("--image: No image path passed!")
                exit()
            background_image = cv2.imread(background_image_loc)
            height, width, _ = background_image.shape
            num_points = -1
            left_hand = convert_csv_to_list_of_tuple(data["Left Hand.x"][0:num_points],
                                                     data["Left Hand.y"][0:num_points], height, width)
            right_hand = convert_csv_to_list_of_tuple(data["Right Hand.x"][0:num_points],
                                                     data["Right Hand.y"][0:num_points], height, width)
            points_dict = create_points_in_radius_dictionary(left_hand + right_hand, radius)
            print("Got points dict...")
            heatmap_img = create_heatmap(points_dict, background_image, len(left_hand), radius)
            print("Finished..")
            cv2.imwrite("Heatmap.png", heatmap_img)

if __name__ == "__main__":
    main()
