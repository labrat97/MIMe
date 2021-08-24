import cv2 as cv
import numpy as np
import syscamera as camera
import vpi
import gc
import time


START_CAP_FRAME = 10
STOP_CAP_FRAME = 20

WINDOW_NAME = 'Disparity'
PVA_NVENC_VIC = vpi.Backend.PVA | vpi.Backend.NVENC | vpi.Backend.VIC


captures = []
retVal = True

for idx in range(2):
    config = camera.configurationString(idx, 1920, 1080, 30)
    captures.append(cv.VideoCapture(config, cv.CAP_GSTREAMER))

frameCache = []
cacheClr: int = 0
frameNum: int = 0

while retVal and len(frameCache) < (STOP_CAP_FRAME-START_CAP_FRAME):
    if cacheClr % 5 == 0:
        gc.collect()
        vpi.clear_cache()
    cacheClr = cacheClr + 1

    frames = []
    rawFrames = []
    for cap in captures:
        ret, frame = cap.read()
        rawFrames.append(frame)
        retVal = (retVal and ret)
        if not retVal:
            break
        with vpi.Backend.CUDA:
            gray = vpi.asimage(frame).convert(vpi.Format.Y16_ER).rescale((1920,1080))
        with vpi.Backend.VIC:
            grayBL = gray.convert(vpi.Format.Y16_ER_BL)
        frames.append(grayBL)
    frameNum = frameNum + 1
    
    frames = frames[::-1]
    rawFrames = rawFrames[::-1]
    confMap = vpi.Image(frames[0].size, vpi.Format.U16)
    disp = vpi.stereodisp(frames[0], frames[1], out_confmap=confMap, backend=PVA_NVENC_VIC, window=5, maxdisp=256)
    dispConv = disp.convert(vpi.Format.U8, backend=vpi.Backend.CUDA, scale=255.0/(32*256))
    dispColor = cv.applyColorMap(dispConv.cpu(), cv.COLORMAP_JET)
    
    confMapConv = confMap.convert(vpi.Format.U8, backend=vpi.Backend.CUDA, scale=255.0/65535)
    mask = cv.threshold(confMapConv.cpu(), 1, 255, cv.THRESH_BINARY)[1]
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    dispColor = cv.bitwise_and(dispColor, mask)


    if retVal:
        if frameNum >= START_CAP_FRAME:
            cv.imwrite(str(frameNum) + ".bmp", dispColor)
            frameCache.append(dispColor)
            print(f'savenum: {frameNum}')
        else:
            print(f'dropnum: {frameNum}')
        
        time.sleep(1./31)        


for cap in captures:
    cap.release()

