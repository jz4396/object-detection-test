import numpy as np
import cv2
import os
import sys
from matplotlib import pyplot as plt
import datetime

try:
    camL = cv2.VideoCapture(0)
    camR = cv2.VideoCapture(1)
    cv2.imshow("Left Camera",camL.read()[1])
    cv2.imshow("Right Camera",camR.read()[1])
except:
    print("Cameras did not load propperly")
    cv2.waitKey(0)
    sys.exit(1)
finally:
    cv2.destroyAllWindows()

fname = os.environ['COMPUTERNAME']+'_rect_Calibration.npz'

try:
    calibData = np.load(fname)
    mtx = calibData['arr_0']
    dist = calibData['arr_1']    
except:
    print("The calibration file "+ fname + " doesn't exist")
    cv2.waitKey(0)
    sys.exit(1)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, camL.read()[1].shape[1::-1], 1, camL.read()[1].shape[1::-1])
x, y, w, h = roi





# stereo = cv2.StereoBM_create(16,13)  #basic version
stereo = cv2.StereoSGBM_create(minDisparity = 0,
        numDisparities = 112-0,
        blockSize = 2,
        P1 = 8*3*3**2,
        P2 = 32*3*3**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 9,
        speckleWindowSize = 50,
        speckleRange = 64,
        mode = cv2.STEREO_SGBM_MODE_HH4
    )

#stereo = cv2.StereoMatcher(minDisparity = 0,
#        numDisparities = 112-0,
#        blockSize = 2,
#        P1 = 8*3*3**2,
#        P2 = 32*3*3**2,
#        disp12MaxDiff = 1,
#        uniquenessRatio = 9,
#        speckleWindowSize = 50,
#        speckleRange = 64,
#        mode = cv2.STEREO_SGBM_MODE_HH4
#    )



h, w = camL.read()[1].shape[:2]
f = 0.8*w                          # guess for focal length
Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w+b') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

while(True):
    dstL = cv2.undistort(camL.read()[1], mtx, dist, None, newcameramtx)
    dstL = dstL[y:y+h, x:x+w]
    dstR = cv2.undistort(camR.read()[1], mtx, dist, None, newcameramtx)
    dstR = dstR[y:y+h, x:x+w]
    #dstR = cv2.cvtColor(dstR, cv2.COLOR_BGR2GRAY)
    #dstL = cv2.cvtColor(dstL, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(dstL,dstR)
    cv2.imshow("Left Camera",dstL)
    cv2.imshow("Right Camera",dstR)
    #disp = cv2.resize(disp,None,fx=.25,fy=.25,interpolation = cv2.INTER_LINEAR)
    #plt.imshow(disp,'gray')
    #plt.pause(0.001)
    cv2.imshow("disp",(disp-disp.min())/(112-32))
    key = cv2.waitKey(1)
    if(key==27):
        break
    elif(key==ord(' ')):
        pts3d = cv2.reprojectImageTo3D(disp,Q)
        color = cv2.cvtColor(dstL,cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = pts3d[mask]
        out_colors = color[mask]
        fn3d = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"-3d.ply"
        write_ply(fn3d, out_points, out_colors)
        print(fn3d," saved")
        cv2.waitKey(0)
    

cv2.destroyAllWindows()
    