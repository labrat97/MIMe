import cv2 as cv
import numpy as np
import syscamera

CHESSBOARD_WIDTH:int = 6
CHESSBOARD_HEIGHT:int = 7
CAMERA_IDX:int = 0
BOARD_SIZE:tuple() = (7,6)
CAMERA_SECOND:int = 21
GOOD_FRAMES:int = 5 * CAMERA_SECOND
PRINT_RESULTS:bool = False
CALIB_FNAME:str = f'cam{CAMERA_IDX}.npz'

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
capture = syscamera.config.setupCamera(CAMERA_IDX)
capret, frame = capture.read()
if not capret:
    exit()

# Try to calibrate while the camera is behaving and the 
while capret:
    # Flatten
    grey = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    cornret, corners = cv.findChessboardCorners(grey, BOARD_SIZE, None)
    
    # Find the sub-pixel coordinates of the corners in the frames
    if cornret:
        resobj.append(oPoints)
        if PRINT_RESULTS: print(f'{len(oPoints)} corners found...')

        # The tuples here are basically magic to me right now:
        # https://docs.opencv.org/4.5.1/dc/dbb/tutorial_py_calibration.html
        subcorn = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)
        resimg.append(subcorn)
    
    # Take picture if time or don't and leave the loop
    if len(resobj) < GOOD_FRAMES:
        capret, frame = capture.read()
    else:
        capret = False

calibret, calibMat, distCoeff, rvecs, tvecs = cv.calibrateCamera(resobj, resimg, grey.shape[::-1], None, None)
imgHeight, imgWidth = grey.shape[:2]
optimalCalibMat, roi = cv.getOptimalNewCameraMatrix(calibMat, distCoeff, (imgWidth,imgHeight), alpha=1, newImgSize=(imgWidth,imgHeight))

if PRINT_RESULTS:
    print(f'Calibration matrix:\t{calibMat}')
    print(f'Distortion coeff\'s:\t{distCoeff}')
    print(f'Optimal calibration matrix:\t{optimalCalibMat}')
    print(f'ROI matrix for crop:\t{roi}')

    meanErr = 0
    for idx in range(len(oPoints)):
        errPoints, _ = cv.projectPoints(oPoints[idx], rvecs[idx], tvecs[idx], calibMat, distCoeff)
        error = cv.norm(resimg[idx], errPoints, cv.NORM_L2)/len(errPoints)
        meanErr += error
    
    print(f'Total error:\t {meanErr/len(oPoints)}')

np.savez(CALIB_FNAME, calib=calibMat, dist=distCoeff, calibOpt=optimalCalibMat, roi=roi)
print(f'Saved to \"{CALIB_FNAME}\"')
