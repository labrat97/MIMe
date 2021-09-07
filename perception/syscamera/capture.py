import cv2 as cv
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

from . import config


class Retina(cv.VideoCapture):
    def __init__(self, id:int, calibFile:str=None, width:int=3280, height:int=2464, fps:int=21, fullBalance:float=0.77):
        super(Retina, self).__init__(
            config.configurationString(camID=id, width=width, height=height, fps=fps),
            cv.CAP_GSTREAMER)
        
        # Assert that the settings were successfully applied
        assert round(self.get(cv.CAP_PROP_FRAME_WIDTH)) == width
        assert round(self.get(cv.CAP_PROP_FRAME_HEIGHT)) == height
        assert round(self.get(cv.CAP_PROP_FPS)) == fps
        
        # Load calibration
        if calibFile is None:
            calibFile = CURRENT_DIR + os.path.sep + f'cam{id}.npz'
        self.calib = np.load(calibFile)
        assert (self.calib['width']/self.calib['height']) == (width/height)

        # Scale K
        self.calibScalar = width/self.calib['width']
        self.__K = self.calib['K'] * self.calibScalar; self.__K[2,2] = 1.

        # Cache harder computations for later
        self.__fullBalance = fullBalance
        DIM = (self.calib['width'], self.calib['height'])
        self.__dimNorm = DIM
        self.__dimOpt = DIM
        self.__computeMaps()

    def __computeMaps(self):
        NORM_BALANCE:float = self.__fullBalance
        OPT_BALANCE:float = 0

        self.__newK = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(self.__K, self.calib['D'], \
            self.__dimNorm, np.eye(3), balance=NORM_BALANCE)
        self.__newKopt = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(self.__K, self.calib['D'], \
            self.__dimOpt, np.eye(3), balance=OPT_BALANCE)

        self.__map0, self.__map1 = cv.fisheye.initUndistortRectifyMap(self.__K, self.calib['D'], \
            np.eye(3), self.__newK, self.__dimNorm, cv.CV_16SC2)
        self.__map0opt, self.__map1opt = cv.fisheye.initUndistortRectifyMap(self.__K, self.calib['D'], \
            np.eye(3), self.__newKopt, self.__dimOpt, cv.CV_16SC2)

    def read(self, undist:bool=True, undistOptimal:bool=False, undistBoth:bool=False):
        # Read from camera
        ret, rawframe = super(Retina, self).read()
        if not ret: return ret, rawframe

        # Requested raw return
        if not undist: return ret, rawframe
        
        # Remap individual frame then return
        if not undistBoth:
            if undistOptimal:
                map0 = self.__map0opt
                map1 = self.__map1opt
            else:
                map0 = self.__map0
                map1 = self.__map1
            frame = cv.cuda.remap(rawframe, map0, map1, \
                interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

            return ret, frame
        
        # Remap both frames then return
        frameOpt = cv.cuda.remap(rawframe, self.__map0opt, self.__map1opt, \
            interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        frameNorm = cv.cuda.remap(rawframe, self.__map0, self.__map1, \
            interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        return ret, frameOpt, frameNorm

class RetinalEmbedder():
    def __init__(self, retina:Retina, outputSize:tuple, peripheralSize:tuple=None):
        assert len(outputSize) == 2
        assert len(peripheralSize) == 2

        self.retina = retina
        self.outputSize = outputSize
        self.peripheralSize = peripheralSize

