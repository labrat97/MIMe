import cv2
import numpy as np
import os

# CONFIG
OUTER_IRIS_RATIO = 1
INNER_IRIS_RATIO = 0.666
IRIS_PADDING = 10
Y_STRIDE = 16
X_STRIDE = 2

__path__ = os.path.dirname(os.path.abspath(__file__))
occupancy = np.genfromtxt(__path__+os.path.sep+'maskOccupancy.csv', delimiter=',').transpose()
mask = cv2.imread(__path__+os.path.sep+'mask.png')

enditer = False
for i in range(occupancy.shape[0]):
    if i % Y_STRIDE != 0: continue
    if np.max(occupancy[i]-IRIS_PADDING) <= 0.0: continue
    if enditer: break

    for j in range(occupancy.shape[1]):
        if j % X_STRIDE != 0: continue
        
        #img = np.zeros((occupancy.shape[0], occupancy.shape[1], 3), np.uint8)
        img = (np.copy(mask)*0.25).astype(np.uint8)
        maxRadius = occupancy[i][j]-IRIS_PADDING
        if maxRadius < 0: continue

        img = cv2.circle(img, (j,i), int(maxRadius*OUTER_IRIS_RATIO), (255,255,255), thickness=-1)
        img = cv2.circle(img, (j,i), int(maxRadius*INNER_IRIS_RATIO), (0,0,0), thickness=-1)

        cv2.imshow('iris', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            enditer = True
            break

cv2.destroyWindow('iris')
