import numpy as np
import cv2
import os
import sys

try:
    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)
    cv2.imshow(camL.read()[1])
    cv2.imshow(camR.read()[1])
except:
    print("Cameras did not load propperly")
    cv2.waitKey(0)
    sys.exit(1)
finally:
    cv2.destroyAllWindows()

fname = os.environ['COMPUTERNAME']+'_rect_Calibration.npz'

try:
    mtx, dist = np.load(fname)
except:
    print("The calibration file "+ fname + " doesn't exist")
    cv2.waitKey(0)
    sys.exit(1)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, camL.shape[1::-1], 1, camL.shape[1::-1])
x, y, w, h = roi
stereo = cv2.StereoBM_create(16,15)

while(cv2.waitKey(1)!=27):
    dstL = cv2.undistort(camL.read()[1], mtx, dist, None, newcameramtx)
    dstL = dstL[y:y+h, x:x+w]
    dstR = cv2.undistort(camR.read()[1], mtx, dist, None, newcameramtx)
    dstR = dstR[y:y+h, x:x+w]
    disp = stereo.compute(imgL,imgR)
    cv2.imshow("ugly",disp)
    
    