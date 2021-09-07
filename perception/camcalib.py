import cv2 as cv
import numpy as np
import sys

CAMERA_IDX:int = 0
PRINT_RESULTS:bool = True
IN_FNAME:str = f'calibcorners{CAMERA_IDX}.npz'
CALIB_FNAME:str = f'cam{CAMERA_IDX}.npz'

if len(sys.argv) > 1:
    IN_FNAME = sys.argv[1]
    print(f'Selected camera corner capture located at: \"{CALIB_FNAME}\"')
if len(sys.argv) > 2:
    CALIB_FNAME = sys.argv[2]
    print(f'Outfile set to: \"{CALIB_FNAME}\"')
params = np.load(IN_FNAME)
print(f'Loaded input corners...')

K = np.zeros((3,3), dtype=np.float64)
D = np.zeros((4,1), dtype=np.float64)
rms, _, _, _, _ = cv.fisheye.calibrate(params['objPoints'], params['imgPoints'], (params['width'], params['height']), K, D, \
    flags=(cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC))

if PRINT_RESULTS:
    print(f'K: \t{K}')
    print(f'D: \t{D}')
    print(f'RMS: \t{rms}')

np.savez(CALIB_FNAME, K=K, D=D, width=params['width'], height=params['height'], rms=rms)
print(f'Saved to \"{CALIB_FNAME}\"')
