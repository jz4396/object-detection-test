import cv2
import numpy as np
import sys

cam = cv2.VideoCapture(0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

try:
    _, cap = cam.read()
    cv2.imshow('',cap)
    cv2.destroyAllWindows()
except:
    print('No Camera')
    cv2.waitKey(0)
    sys.exit(1)

board_size = (5,6)

brdp = np.zeros((np.prod(board_size),3), np.float32)
brdp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)

brdpts = []
imgpts = []

while(True):
    _, cap = cam.read()
    
    exist, corners = cv2.findChessboardCorners(cap,board_size,None)
    cornered = cv2.drawChessboardCorners(cap,board_size,corners,exist)
    cv2.imshow('cap',cap)
    key = cv2.waitKey(1)
    if(key == 27) :
        break
    elif(key == ord(" ") and exist):
        print('--------------------------------------Saving---------------------------------------')
        brdpts.append(brdp[:])
        ac_corners = cv2.cornerSubPix(cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), criteria)
        cornered = cv2.drawChessboardCorners(cap,board_size,ac_corners,exist)
        imgpts.append(ac_corners[:])
        print(len(brdpts),',',len(imgpts))
        cv2.waitKey(10)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(brdpts, imgpts, cap.shape[1::-1], None, None)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, cap.shape[1::-1], 1, cap.shape[1::-1])

print(roi)

mean_error = 0
for i in range(len(brdpts)):
    imgpoints2, _ = cv2.projectPoints(brdpts[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpts[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(brdpts)) )

while(cv2.waitKey(1)==-1):
    dst = cv2.undistort(cam.read()[1], mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    try:
        cv2.imshow('rec',dst)
    except:
        print('Bad Dataset')
        cv2.waitKey(0)
        sys.exit(1)

cv2.destroyAllWindows

print('Press space to save, anything else to exit')

if(cv2.waitKey(0)==ord(' ')):
    print('Saving!!!!!')
    np.savez(os.environ['COMPUTERNAME']+'_rect_Calibration',mtx,dist)
    print('Calibration Saved')