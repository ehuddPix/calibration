

# the purpose of this program is to prepare calibration for a single camera  by producing the calibration matrix and the distortion matrix 
# input the images from each camera are being red by the program and a JSON file is saved with the calculated results 
# this JSON file will later be used as an input for the stereo calibration 


import yaml
import cv2
import numpy as np
import glob
import time
import json

def current_milli_time():
    return round(time.time() * 1000)

print ( cv2.__version__[0])#4.3.5
debug=False
CHECKERBOARD = (13,24)#(8,13)  13 width 24 height ,  the calibration during the stereo stage ( next stage  i stereoCalibration.py ) will be done holding  the board on its narrow side 
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) # didnt find much benefit in increasing this process eg 300, 0.001 

# note: these are the calibration flags you can unfix all the parameters m K1--K4
calibration_flags =cv2.fisheye.CALIB_FIX_SKEW+cv2.fisheye.CALIB_FIX_K1+cv2.fisheye.CALIB_FIX_K2+cv2.fisheye.CALIB_FIX_K3+cv2.fisheye.CALIB_FIX_K4+cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC 


objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
raw_images= []
images = glob.glob('*Left*.png')#here we load the image when we load the right camera  image set we shall load with *Right*.png  as we use "Left" and "Right" as naming convention 
for image in images:
    print(image)
    img = cv2.imread(image)
    imgraw = cv2.imread(image)
    raw_images.append(imgraw)
    cv2.imshow('candidate gray', img)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('candidate gray', gray)
    
    # Chess board corners
    start_time=current_milli_time()
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    end_time =current_milli_time()
    print(str(end_time-start_time))
    # Image points (after refinin them)
    if ret == True:
        objpoints.append(objp)     
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('candidate gray', img)
        cv2.waitKey(10)
    else: print('bad image,failed chessboard='+image)
 
# so far we have collected the  image points(2D ) and we pushed a 3D point that represent the chessboard corners in 3D space 
      
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, rvecs, tvecs = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print('root mean square error :  rms='+str(rms))
if debug:input("rms")
#print(rvecs)
#print(tvecs)
if debug:input("press")
print("Found " + str(N_OK) + " valid images for calibration")
DIM=_img_shape[::-1]
balance=1
dim2=None
dim3=None


img = cv2.imread("a.png")
dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

if debug:input ("wait")  
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

data = {'dim1': dim1, 
        'dim2':dim2,
        'dim3': dim3,
        'K': np.asarray(K).tolist(), 
        'D':np.asarray(D).tolist(),
        'new_K':np.asarray(new_K).tolist(),
        'scaled_K':np.asarray(scaled_K).tolist(),
        'balance':balance}


with open("fisheye_calibration_data.json", "w") as f:
    json.dump(data, f)

cv2.imshow("undistorted", undistorted_img)
#img2 = cv2.imread("aa.png")
#cv2.imshow("none undistorted", img2)
errors_per_set=[]
for i in range(len(objpoints)): 
            # we have the camera model and the transformation matrix per chessBoard image so we  we take the dimensions of the chessBoard and transform them back to the image 
            #thus we witness if they are projected properly 
            imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)                                    
            errors_per_image=[]
            image_i = raw_images[i]
            image_i_raw = image_i.copy()
         
            for k in range (len(imgpoints[0]) ):               
                center_coordinates = (int(imgpoints2[0][k][0]),int( imgpoints2[0][k][1]))
                center_coordinates_imgpoints = (int(imgpoints[i][k][0][0]),int(imgpoints[i][k][0][1]))
                color = (255, 0, 0)
                color2 = (0, 255, 0)             
                error = cv2.norm(center_coordinates_imgpoints, center_coordinates, cv2.NORM_L2)               
                x1, y1 = center_coordinates[0], center_coordinates[1]
                x2, y2 = center_coordinates_imgpoints[0]+5*int(error),center_coordinates_imgpoints[1]+5*int(error)
                errors_per_image.append(error)
                errors_per_set.append(error)
                #print(error)
                line_thickness = 3
                color = (0, 255, 0)
                if (error>1) : 
                    color=(0,0,180+int(error*10))                                    
                font = cv2.FONT_HERSHEY_SIMPLEX               
                #incase we want to debug we can plot the reprojected points on top of the projected one with  
                #prporional shift as per  the error 
                #cv2.putText(image_i, 'mean ='+str(mean)+' var='+ str(variance), (100,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                image_i= cv2.line(image_i,(x1, y1), (x2, y2), color, thickness=line_thickness)
                image_i = cv2.circle(image_i,center_coordinates, 10, color, 1)
                image_i = cv2.circle(image_i,center_coordinates_imgpoints, 5, color2, 1)               
            #calculate avg error per image 
            #calculate variance
            mean  = np.mean(errors_per_image)
            variance  = np.var(errors_per_image)
            #print(errors_per_image)
            print (mean)
            print (variance)           
            # we plot the error as part of the file name so we can filter out the images with high RMS , given that they do not cover unique angles and
            #dont harm too much the homogenous distribution of angles distances of chessboard images from the camera 
            cv2.imwrite('_'+str(error)+'image_'+str(i)+'_.png', image_i_raw)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_i, 'mean ='+str(mean)+' var='+ str(variance), (100,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            #incase we want to plot the data on the images , 
            #cv2.putText(image_i, 'mean ='+str(mean), (100,250), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #cv2.imshow("image", image_i)
            #cv2.imwrite('__'+str(error)+'image_'+str(i)+'_.jpg', image_i)
            #error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)                     
total_mean =  np.mean(errors_per_set)
total_variance =  np.var(errors_per_set)
cv2.putText(undistorted_img, 'mean ='+str(total_mean)+' var='+ str(total_variance), (100,100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite('-undistorted_img.jpg' , undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

 
