import cv2 as cv
import numpy as np
import syscamera
import sys

CAMERA_IDX:int = 0
if len(sys.argv) > 1:
    CAMERA_IDX = int(sys.argv[1])
    print(f'Selected camera index {CAMERA_IDX} for capture...')

CHESSBOARD_WIDTH:int = 9
CHESSBOARD_HEIGHT:int = 6
BOARD_SIZE:tuple() = (CHESSBOARD_HEIGHT,CHESSBOARD_WIDTH)
CAMERA_SECOND:int = 21
CAP_FRAMES:int = CAMERA_SECOND*6
PRINT_RESULTS:bool = True
CALIB_FNAME:str = f'calibcorners{CAMERA_IDX}.npz'
DIM_SCALAR:float = 0.5

# Set up the termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

# Prepare object points according to:
# https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
oPoints = np.zeros((1, CHESSBOARD_HEIGHT*CHESSBOARD_WIDTH, 3), np.float32)
oPoints[0,:,:2] = np.mgrid[0:CHESSBOARD_HEIGHT, 0:CHESSBOARD_WIDTH].T.reshape(-1, 2)

# Arrays to store the resulting points
resobj = [] # World-space, 3D space
resimg = [] # 2D image coordinates

# Open the interface to pull frames from the selected camera
capture = syscamera.config.setupCamera(CAMERA_IDX)
capret, frame = capture.read()
if not capret:
    exit()

# Try to calibrate while the camera is behaving and the capture is opened
frames = []
while capret and capture.isOpened():
    # Flatten
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    size = tuple([int(dim*DIM_SCALAR) for dim in grey.shape])
    grey = cv.resize(grey, size, cv.INTER_LINEAR_EXACT)
    if PRINT_RESULTS: print(f'Captured frame {len(frames)} and converted to grey...')
    frames.append(grey)
    
    # Take picture if time or don't and leave the loop
    if len(frames) < CAP_FRAMES:
        capret, frame = capture.read()
    else:
        capret = False
capture.release()
if PRINT_RESULTS: print('Capture released for analysis.')

for idx, frame in enumerate(frames):
    cornret, corners = cv.findChessboardCorners(frame, BOARD_SIZE, None, \
        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_FILTER_QUADS + \
        cv.CALIB_CB_ACCURACY + cv.CALIB_CB_NORMALIZE_IMAGE)
    if PRINT_RESULTS: print(f'Corner computation {idx} completed...')
    
    # Find the sub-pixel coordinates of the corners in the frames
    if cornret:
        resobj.append(oPoints)
        if PRINT_RESULTS: print(f'{oPoints.shape[1]} corners found...')

        # The tuples here are basically magic to me right now:
        # https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
        subcorn = cv.cornerSubPix(grey, corners, (3,3), (-1,-1), criteria)
        resimg.append(subcorn)

np.savez(CALIB_FNAME, imgPoints=resimg, objPoints=resobj, width=grey.shape[1], height=grey.shape[0])
print(f'Corners saved to: \"{CALIB_FNAME}\"')
