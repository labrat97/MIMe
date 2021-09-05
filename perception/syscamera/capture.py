import cv2 as cv
import numpy as np
import torch

from . import config

class RetinalSystem(cv.VideoCapture):
    def __init__(self, id:int, calib:np.ndarray, width:int=3280, height:int=2464, fps:int=21):
        super(RetinalSystem, self).__init__(
            config.configurationString(camID=id, width=width, height=height, fps=fps),
            cv.CAP_GSTREAMER)
        
        assert calib is not None
        self.intrinsic = calib

    def read(self, undist:bool=True, undistOptimal:bool=False):
        rawFrame, ret = super(RetinalSystem, self).read()
        if not ret: return ret

        
