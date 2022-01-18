# the purpose of this program is to prepare calibration for a stereo  camera  by producing the rotation matrix and the transformation  matrix 
# between the first and the second cmaera that are being inserted to the stereoCalibration function 
# we use rodrigues vector 
# input the images that are seen by both cameras are being red pair after pair and eventually the stereoCalibration use all the pairs to 
# figure out the right rotation and translation matrix 
# 
# we suppose to take them and provide transformation and rotation matrices in the pixellot format 
#


import math  
import json
import os
import cv2
import numpy as np
import glob
import re
import time
import matplotlib.pyplot as plt

numbers = re.compile(r'(\d+)(\d+)(\d+)i')

def markPoints( Color , img , setOfPoints):
    for point in setOfPoints:
        center_coordinates = [int(point[0][0]),int( point[0][1])]       
        #print (center_coordinates)
        # project them on the left image and see if they land properly 
        color = Color       
        markedImg = cv2.circle(img,center_coordinates, 10, color, 3)
    return markedImg
    
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    #img = cv.line(img,np.int0(corner),(120,200),(0,0,255),2)
    img = cv2.line(img, np.int0(corner), tuple(np.int0(imgpts[0].ravel())), (255,0,0), 2)
    img = cv2.line(img, np.int0(corner), tuple(np.int0(imgpts[1].ravel())), (0,255,0),2)
    img = cv2.line(img, np.int0(corner), tuple(np.int0(imgpts[2].ravel())), (0,0,255), 2)
    return img

def yawpitchrolldecomposition(R):

    sin_x    = math.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
    validity  = sin_x < 1e-6
 
    z1    = math.atan2(R[2,0], R[2,1])     # around z1-axis
    x      = math.atan2(sin_x,  R[2,2])     # around x-axis
    z2    = math.atan2(R[0,2], -R[1,2])    # around z2-axis
   
    return np.array([[z1], [x], [z2]])




def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def current_milli_time():
    return round(time.time() * 1000)

def scale(scale_percent,image): 
	 width = int(image.shape[1] * scale_percent / 100)
	 height = int(image.shape[0] * scale_percent / 100)
	 dim = (width, height)
	 resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	 return resized 

# Axis i use it in debug purposes to project lines in space .  
axis = np.float32([[3.4214,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3) , 
# Visualization options
drawCorners = False
showSingleCamUndistortionResults = True
showStereoRectificationResults = True
writeUdistortedImages = True
imageToDisp = './scenes/scene_1280x480_1.png'
# Calibration settings
CHECKERBOARD = (13,24)#(13,8)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objp[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

_img_shape = None
objpointsLeft = [] # 3d point in real world space
imgpointsLeft = [] # 2d points in image plane.

objpointsRight = [] # 3d point in real world space
imgpointsRight = [] # 2d points in image plane.
objectPoints=[]
# for each camera( right and left ) load the K , D  data 
# Opening JSON file for left 
f = open('fisheye_calibration_data_LEFT.json')
dataL = json.load(f)
KL = np.zeros((3, 3))
DL = np.zeros((4, 1))
KL=np.array(dataL['K'], dtype=np.float64)
DL=np.array(dataL['D'], dtype=np.float64)

#opening jso file for right 

f2 = open('fisheye_calibration_data_RIGHT.json') 
dataR = json.load(f2)

KR= np.zeros((3, 3))
DR = np.zeros((4, 1))
KR=np.array(dataR['K'], dtype=np.float64)
DR=np.array(dataR['D'], dtype=np.float64)

# read a pair of images 
imagesL= glob.glob('*Left*.png')#'imageL*.png'
imagesL=sorted(imagesL, key=numericalSort)
imagesR=sorted(glob.glob('*Right*.png'), key=numericalSort)#imagesR=sorted(filter(os.path.isfile,   glob.glob('*Right*.png')))  #just another valid and tested option 

# Find the chessboard corners
for imgageL in imagesL:
    imgl = cv2.imread(imgageL)
    grayL = cv2.cvtColor(imgl,cv2.COLOR_BGR2GRAY)
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    objpointsLeft.append(objp)
    cv2.cornerSubPix(grayL,cornersL,(3,3),(-1,-1),subpix_criteria)
    imgpointsLeft.append(cornersL) 
    

for imageR in imagesR:
    imgr = cv2.imread(imageR)
    grayR = cv2.cvtColor(imgr,cv2.COLOR_BGR2GRAY)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    objpointsRight.append(objp)
    cv2.cornerSubPix(grayR,cornersR,(3,3),(-1,-1),subpix_criteria)
    imgpointsRight.append(cornersR)
    if retR == True:  # i know that retL woll also be ok since i filteres out all the images 
        objectPoints.append(objp)
        

# after the data is collected push it to the stereo-claibration 
# Stereoscopic calibration

    # We need a lot of variables to calibrate the stereo camera
"""
Based on code from:
https://gist.github.com/aarmea/629e59ac7b640a60340145809b1c9013
"""
processing_time01 = cv2.getTickCount()

K_left = np.zeros((3, 3), dtype=np.float64)
D_left = np.zeros((4, 1), dtype=np.float64)

K_right = np.zeros((3, 3), dtype=np.float64)
D_right = np.zeros((4, 1), dtype=np.float64)

R = np.zeros((1, 1, 3), dtype=np.float64)
T = np.zeros((1, 1, 3), dtype=np.float64)

imageSize= (3840, 2160)

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
OPTIMIZE_ALPHA = 0.25

print("Calibrating cameras together...")
#print(DR)

# Stereo calibration

print('---')
#print(imgpointsLeft)

objectPoints = np.asarray(objectPoints, dtype=np.float64)
imgpointsLeft= np.asarray(imgpointsLeft, dtype=np.float64)
imgpointsRight= np.asarray(imgpointsRight, dtype=np.float64)
(RMS, K_left, D_left, K_right, D_right, R, T) = cv2.fisheye.stereoCalibrate(
        objectPoints, imgpointsLeft, imgpointsRight,
        KL, DL,
        KR, DR,
        imageSize, None, None,
        cv2.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)
#
# Print RMS result (for calibration quality estimation)

print ("<><><><><><><><><><><><><><><><><><><><>")
print ('rotation')
print ( R )
print ( 'translation' )
print (T)
print ("<><>   RMS is ", RMS, " <><>")
print ("<><><><><><><><><><><><><><><><><><><><>")    
print("Rectifying cameras...")

print(" rodriges  ")
rotV, _ = cv2.Rodrigues(R)
#rotV = rotV*180/math.pi
print (rotV)
roll = 180*math.atan2(-R[2][1], R[2][2])/math.pi
pitch = 180*math.asin(R[2][0])/math.pi
yaw = 180*math.atan2(-R[1][0], R[0][0])/math.pi
print("yaw")
print ( yaw)
print("pitch")
print ( pitch)
print("roll")
print(roll)
print (yawpitchrolldecomposition(R))
'''
imgtestoutput= cv2.warpPerspective(imgtest,rotV,(300,300))#cv2.warpAffine(imgtest,rotV,(300,300)) #
 #draw_axis(img, R, t, K):
#imgtestoutput = draw_axis(imgtest, R, T, KL)
plt.subplot(121),plt.imshow(imgtest),plt.title('Input')
plt.subplot(122),plt.imshow(imgtestoutput),plt.title('Output')
plt.show()
'''

R1 = np.zeros([3,3])
R2 = np.zeros([3,3])
P1 = np.zeros([3,4])
P2 = np.zeros([3,4])
Q = np.zeros([4,4])

# Rectify calibration results
(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap) = cv2.fisheye.stereoRectify(
                KL, DL,
                KR, DR,
                imageSize, R, T,
                0, R2, P1, P2, Q,
                cv2.CALIB_ZERO_DISPARITY, (0,0) , 0, 0)
                      
# Saving calibration results for the future use
print("Saving calibration...")
leftMapX, leftMapY = cv2.fisheye.initUndistortRectifyMap(
        KL, DL, leftRectification,
        leftProjection, imageSize, cv2.CV_16SC2)

rightMapX, rightMapY = cv2.fisheye.initUndistortRectifyMap(
        KR, DR, rightRectification,
        rightProjection, imageSize, cv2.CV_16SC2)

imgLTest = cv2.imread('TEST_501_left.png')
imgRTest = cv2.imread('TEST_501_right.png')
grayL = cv2.cvtColor(imgLTest,cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgRTest,cv2.COLOR_BGR2GRAY)
 # Rectifying left and right images

imgR= cv2.remap( imgRTest, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
imgL = cv2.remap(imgLTest, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

grayL_remap= cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
grayR_remap = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
 




 

# extract the corner points form Right  image
retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
cv2.cornerSubPix(grayR_remap,cornersR,(6,6),(-1,-1),subpix_criteria)

retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
cv2.cornerSubPix(grayL_remap,cornersL,(6,6),(-1,-1),subpix_criteria)

objptst = np.zeros( (CHECKERBOARD[0]*CHECKERBOARD[1], 1, 3) , np.float64)
objptst[:,0, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Find the rotation and translation vectors.
# we deduce rotation and translation vectors by solving the transformatin needed to project 3D pattern to known 2d pattern in the image  
_,rvec_r,tvec_r,_= cv2.solvePnPRansac(objp, cornersR, K_right,D_right)
#ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
_,rvec_l,tvec_l,_= cv2.solvePnPRansac(objp, cornersL, K_left,D_left)
# project 3D points to image plane
rp_l,_ = cv2.fisheye.projectPoints(objptst, rvec_l, tvec_l, KL,DL)#K_left,D_left)
rp_r,_ = cv2.fisheye.projectPoints(objptst, rvec_r, tvec_r, KR,DR)#K_left,D_left)

# tryng to see if the points can be casted from one camera to the next 
 # calculate world <-> cam2 transformation
print("cv2.Rodrigues(R)[0]")
print(cv2.Rodrigues(R)[0])
rodrig  = cv2.Rodrigues(R)[0]
print(rodrig)
rvec_r2l, tvec_r2l  = cv2.composeRT(rvec_l,tvec_l,rodrig,T)[:2]
# compute reprojection error for cam2
rp_r,_ = cv2.fisheye.projectPoints(objptst, rvec_r2l, tvec_r2l, KR,DR)#K_left,D_left)   
axisPoints, jac = cv2.projectPoints(axis,rvec_r,tvec_r, K_right,D_right)
imgRTest = draw(imgRTest,cornersR,axisPoints)
markedImgR = markPoints((0, 133, 0), imgRTest, rp_r)
markedImg = markPoints((255, 133, 0), imgLTest, rp_l)    
markedImg = markPoints((0, 0, 255), imgLTest, cornersL)    
imgLscaled = scale (20,imgL)
imgRscaled = scale (20,imgR)  
markedImg =  scale (30,markedImg) 
markedImgR =  scale (30,markedImgR)   
cv2.imshow('imgLTest',markedImg)   
cv2.imshow('imgRTestR',markedImgR)   

# https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 11
min_disp = -128
max_disp = 128
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 5
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 200
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 0

#it helps to see the disparity map and the rectified images in case of bed calibration they will appear asymmetrical  

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)
disparity_SGBM = stereo.compute(imgL, imgR)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
disparity_SGBM  = scale(30,disparity_SGBM)
cv2.imshow("Disparity", disparity_SGBM)
cv2.imwrite("disparity_SGBM_norm.png", disparity_SGBM)

cv2.imshow('Left  STEREO CALIBRATED', imgLscaled)
cv2.imshow('Right STEREO CALIBRATED', imgRscaled)
cv2.waitKey(0)
cv2.destroyAllWindows()


