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





# SGBM Parameters -----------------
window_size = 10                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
 
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=192,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)



# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


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

    displ = left_matcher.compute(dstL, dstR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(dstR, dstL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, dstL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    disp = np.uint8(filteredImg)    

    cv2.imshow("Left Camera",dstL)
    cv2.imshow("Right Camera",dstR)
    #disp = cv2.resize(disp,None,fx=.25,fy=.25,interpolation = cv2.INTER_LINEAR)
    #plt.imshow(disp,'gray')
    #plt.pause(0.001)
    cv2.imshow("disp",disp) #(disp-disp.min())/(112-32))
    key = cv2.waitKey(1)
    if(key==27):
        break
    elif(key==ord(' ')):


        h, w = disp.shape[:2]
        f = 0.8*w                          # guess for focal length
        Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])

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
    