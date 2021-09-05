import cv2 as cv
import numpy as np
import syscamera
import os

CHESSBOARD_WIDTH:int = 6
CHESSBOARD_HEIGHT:int = 7
CAMERA_IDX:int = 0

# Set up the termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 60, 0.001)

# Prepare object points according to:
# https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
oPoints = np.zeros((CHESSBOARD_HEIGHT*CHESSBOARD_WIDTH, 3), np.float32)
oPoints[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store the resulting points
resobj = [] # World-space, 3D space
resimg = [] # 2D image coordinates

# Open the interface to pull frames from the selected camera
camConfig = syscamera.configurationString(CAMERA_IDX)
capture = cv.VideoCapture(camConfig, cv.CAP_GSTREAMER)

