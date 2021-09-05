import cv2 as cv
import numpy as np

def configurationString(camID:int, width:int, height:int, fps:int) -> str:
    return f'nvarguscamerasrc sensor_id={camID} ! ' \
        + f'video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}' \
        + f', format=(string)NV12, framerate=(fraction){fps}/1 ! ' \
        + 'nvvidconv ! video/x-raw, format=(string)I420 ! ' \
        + 'videoconvert ! video/x-raw, format=(string)BGR ! ' \
        + 'appsink'

def setupCamera(camID:int, width:int=3264, height:int=2464, fps:int=21) -> cv.VideoCapture:
    config = configurationString(camID=camID, width=width, height=height, fps=fps)
    return cv.VideoCapture(config, cv.CAP_GSTREAMER)

def loadCalibration(path:str) -> np.ndarray:
    result = np.load(path)
    assert result.shape == (3, 3)

    return result
