import cv2 as cv

def configurationString(camID: int, width: int, height: int, fps: int):
    return f'nvarguscamerasrc sensor_id={camID} ! ' \
        + f'video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}' \
        + f', format=(string)NV12, framerate=(fraction){fps}/1 ! ' \
        + 'nvvidconv ! video/x-raw, format=(string)I420 ! videoconvert ! ' \
        + 'video/x-raw, format=(string)BGR ! appsink'

