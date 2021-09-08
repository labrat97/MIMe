import cv2 as cv
import numpy as np
import syscamera
import sys

CAMERA_IDX:int = 0
if len(sys.argv) > 1:
    CAMERA_IDX = int(sys.argv[1])
    print(f'Selected camera index {CAMERA_IDX} for capture...')

CAMERA_SECOND:int = 21
CAPTURE_TIME:int = CAMERA_SECOND*CAMERA_SECOND
VID_FNAME:str = f'cam{CAMERA_IDX}cap.avi'

capture = syscamera.config.setupCamera(CAMERA_IDX)
width:int = int(capture.get(3)/2)
height:int = int(capture.get(4)/2)
size = (width, height)
writer = cv.VideoWriter(VID_FNAME, cv.VideoWriter_fourcc(*'MJPG'), 21, frameSize=size)
frames = np.zeros([CAPTURE_TIME, height, width, 3], dtype=np.uint8)
framenum = 0

while framenum < CAPTURE_TIME:
    ret, frame = capture.read()
    if not ret: break
    frame = cv.resize(frame, size, interpolation=cv.INTER_LINEAR)

    frames[framenum] = frame
    if framenum % int(CAMERA_SECOND/2) == 0:
        print(float(framenum)/CAMERA_SECOND)

    framenum += 1
capture.release()

for idx in range(frames.shape[0]):
    writer.write(frames[idx])
writer.release()

print(f'The video was successfully saved to fname: \"{VID_FNAME}\"')
