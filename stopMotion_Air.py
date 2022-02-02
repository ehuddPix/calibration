
'''
purpose :
    record two  images from the left or right cameras and store them in a file
    we are supposed to collect about 30-50 images where each image is seen by both cameras;
    the calibration chessboard should be circled in two half-spheres,
    each taken at different distances for the cameras. The first is about 2 meters, the second is about 3-4 meters.
    The images should be tilted in 3dof in about 20 degrees on each axis , the
    later they will be processed by the stereocalibration.py
    
Usage: just run the script while the board in front of the cameras,
look at the images taken if the chess is seen correctly and the message mages saved!!!!!!!!!!!!!!!!!!   appears

NOTE :
1) through-out  all the processes of taking pictures, the  cameras should not move

2)Use to fetch images from the air camera head via rtsp protocol when the camera is transmitting the data over WIFI on a local wifi network  (the PC is disconnected from the LAN) and is connected to the air head hotspot) At this stage, we are connected to one camera at a time .Note that there might be delays between the cameras, so the board should stay static while taking pictures 

'''

import cv2
import time 
#cap = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(2)
#time.sleep(5)
CHECKERBOARD = (13,24)
cap = cv2.VideoCapture('rtsp://192.168.3.88')
cap2 = cv2.VideoCapture('rtsp://192.168.2.89') 
num = 1

def scale(scale_percent,image): 
	 width = int(image.shape[1] * scale_percent / 100)
	 height = int(image.shape[0] * scale_percent / 100)
	 dim = (width, height)
	 resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	 return resized 

def current_milli_time():
    return round(time.time() * 1000)
startTime= current_milli_time()

if  cap.isOpened():
    succes1, img = cap.read()  
    cv2.waitKey(1)
    succes2, img2 = cap2.read()
    print ("waitKey")
    
    #if cv2.waitKey(5) & 0xFF == 27:
      #break
    
    
       
    
    if (current_milli_time()-startTime>10):
        
        print(current_milli_time()-startTime)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret1, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        print (ret1)
        gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        print(ret2)
        if ret1 and ret2:
    
            cv2.imwrite('cimages/both/image_' + str(num)+'_Right_'+str(current_milli_time())+ '.png', img2)
            print("savedL")
            cv2.imwrite('cimages/both/image_' + str(num)+'_Left_'+str(current_milli_time())+'.png', img)
            print("images saved!!!!!!!!!!!!!!!!!!")
            startTime =current_milli_time()
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret1)
        cv2.drawChessboardCorners(img2, CHECKERBOARD, corners2, ret2)
     
    img = scale(50,img)
    img2 = scale(50,img2)
    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)
    cv2.waitKey(0)
    
    #cv2.imwrite('cimages/stereoRight/imageR' + str(num) + '.png', img2)
    
    num += 1    
    #time.sleep(3)
    #cv2.imshow('Img 2',img2)

# Release and destroy all windows before termination
cap.release()
cap2.release()

cv2.destroyAllWindows()

