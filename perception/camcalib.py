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
GOOD_FRAMES:int = CAMERA_SECOND*2
CAP_FRAMES:int = CAMERA_SECOND*10
PRINT_RESULTS:bool = True
CALIB_FNAME:str = f'cam{CAMERA_IDX}.npz'
COMPUTE_SCALAR:float = 2

# Set up the termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.00033)

# Prepare object points according to:
# https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
oPoints = np.zeros((CHESSBOARD_HEIGHT*CHESSBOARD_WIDTH, 3), np.float32)
oPoints[:, :2] = np.mgrid[0:CHESSBOARD_HEIGHT, 0:CHESSBOARD_WIDTH].T.reshape(-1, 2)

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
    size = tuple([int(dim/COMPUTE_SCALAR) for dim in grey.shape])
    grey = cv.resize(grey, size, cv.INTER_LINEAR_EXACT)
    if PRINT_RESULTS: print(f'Captured frame {len(frames)} and converted to grey...')
    frames.append(grey)
    
    # Take picture if time or don't and leave the loop
    if cv.waitKey(int(1000/21)) == 27:
        break
    if len(frames) < CAP_FRAMES:
        capret, frame = capture.read()
    else:
        capret = False
capture.release()
if PRINT_RESULTS: print('Capture released for analysis.')

for idx, frame in enumerate(frames):
    cornret, corners = cv.findChessboardCorners(frame, BOARD_SIZE, None, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FILTER_QUADS + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_ACCURACY)
    if PRINT_RESULTS: print(f'Corner computation {idx} completed...')
    
    # Find the sub-pixel coordinates of the corners in the frames
    if cornret:
        resobj.append(oPoints)
        if PRINT_RESULTS: print(f'{len(oPoints)} corners found...')

        # The tuples here are basically magic to me right now:
        # https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
        subcorn = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)
        resimg.append(subcorn)
    
    # Break early
    if len(resimg) >= GOOD_FRAMES:
        break

calibret, calibMat, distCoeff, rvecs, tvecs = cv.calibrateCamera(resobj, resimg, (grey.shape[::-1]), None, None)
imgHeight, imgWidth = grey.shape[:2]
optimalCalibMat, roi = cv.getOptimalNewCameraMatrix(calibMat, distCoeff, (imgWidth,imgHeight), alpha=1, newImgSize=(imgWidth,imgHeight))

calibMat = COMPUTE_SCALAR * calibMat; calibMat[2,2] = 1.
optimalCalibMat = COMPUTE_SCALAR * optimalCalibMat; optimalCalibMat[2,2] = 1.
roi = np.array([COMPUTE_SCALAR * n for n in roi])
size = COMPUTE_SCALAR * np.array([imgWidth, imgHeight], dtype=np.uint32)

if PRINT_RESULTS:
    print(f'Calibration matrix:\t{calibMat}')
    print(f'Distortion coeff\'s:\t{distCoeff}')
    print(f'Optimal calibration matrix:\t{optimalCalibMat}')
    print(f'ROI matrix for crop:\t{roi}')

    meanErr = 0
    for idx in range(len(resobj)):
        errPoints, _ = cv.projectPoints(resobj[idx], rvecs[idx], tvecs[idx], calibMat, distCoeff)
        error = cv.norm(resimg[idx], errPoints, cv.NORM_L2)/len(errPoints)
        meanErr += error
    
    print(f'Total error:\t {meanErr/len(oPoints)}')

np.savez(CALIB_FNAME, calib=calibMat, dist=distCoeff, calibOpt=optimalCalibMat, roi=roi, size=size)
print(f'Saved to \"{CALIB_FNAME}\"')
