# Introduction

This repository is part of my Summer Research Project. The goal of the project is explore the possibility to detect Auslan using the Zipf-Mandelbrot-Li Fast Entropy Function. In order to use the function I would need to first symbolize the data. This repository contains code to symbolize the movement of a signer using Mediapipe's pose estimation.

# Method

Of the 3 python files, the one named `video_grid_symbolizer.py` is the most important one. The other 2 only capture vertical movement, while this one captures both horizontal and vertical movement. `video_grid_symbolizer.py` Divides a person into 6 regions, each region is assigned a letter. At every frame mediapipe is used to estimate the person's pose, and the landmarks for the sholder, hip, left and right hand are collected, the region each hand is in is determined and the frame number, left and right hand symbol, and the label for that frame (whether it's signing or non-signing) is recorded. At the end of the video the recorded data is stored into a file.

# Video Limitations

In order to speed up the process of symbolizing, only the signer must be in the video. This way MRCNN and SORT don't need to be used to segment and track individuals within the video.

# Usage

Using the `video_grid_symbolizer.py` script is straightforward.

```
python3 video_grid_symbolizer.py <path to video files> <output folder path> <groundtruth file> [--vis --output]
```

The `--vis` option will output every frame with the regions drawn. The hands are represented as dots and their color represents the region its in.

# Dependencies
Python Version `3.8.8`

```
pytictoc
numpy
opencv
mediapipe
```
