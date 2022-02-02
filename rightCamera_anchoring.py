#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:05:56 2022

@author: ubuntu20
"""

import numpy as np
import json
import glob
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

'''
purpose : the code below takes  a rotation matrix taken from stereocalibration for fish eye  and an anchor point of the left camera 
usage : 
    insert the rotation matrix and the translation vector  as the left camera anchor point at 
    
the following code draw from the concept presented at : 
    
Output rotation matrix. Together with the translation vector T, 
this matrix brings points given in the first camera's 
coordinate system to points in the second camera's coordinate system. 
In more technical terms, the tuple of R and T performs a change of basis from the 
first camera's coordinate system to the second camera's coordinate system. Due to its duality, 
this tuple is equivalent to the position of the first camera with respect to the second camera coordinate system. 

(https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5)
'''
# initial values 
leftCameraTranslation = [9,11,-3]# position im meters , anchoring the left camera 
testVector=[0,0,10]#[10,20,30]

chessSquareSize_milimeters  = 45 # here the chess square of the calibration is 45 mm 
scale = -chessSquareSize_milimeters/1000# the shifting to the right camera at a scale of meters translated from milimiters 

#this is the translation vector 
translation_calibration = [-1.7977488*scale,-0.03258579*scale,-1.62800927*scale]
deltaTranslation=translation_calibration #translation_calibration

# this is the matrix taken from calibration scaled to meters 
rotationMatrix_R = R.from_matrix ([
 [ 5.83785996e-04  ,-1.74835793e-02, -9.99846980e-01],
 [-4.55236175e-03 , 9.99836744e-01 , -1.74860583e-02],
 [ 9.99989468e-01 ,4.56187326e-03 ,5.04099112e-04 ]])


# common rotation matrix for draft calculations 
rotationZero = R.from_matrix([[1,0,0,],
                               [0, 1,0],
                               [0,0,1]])
 

rotate_Z_CC_45 = R.from_matrix([[0.707,0.707,0],
                               [-0.707,  0.707  , 0],
                               [0,0,1]
                         ])

rotate_Z_90 = R.from_matrix([[0, -1, 0],

                             [1, 0, 0],

                             [0, 0, 1]])

rotate_X_90 = R.from_matrix([[1, 0, 0],

                             [0, 0,-1],

                             [0, 1, 0]])

rotate_CC_X_90 = rotate_X_90.inv()


rotate_Y_90 = R.from_matrix([[0, 0, 1],

                             [0, 1,0],

                             [-1, 0, 0]])

# rotate 45 degrees countrer clockwise
rotate_Y_CC_45= R.from_matrix([[np.cos(np.pi/4), 0, -np.sin(np.pi/4)],

                             [0, 1,0],

                             [np.sin(np.pi/4), 0, np.cos(np.pi/4)]])


rotate_Y_CC_90= R.from_matrix([[np.cos(np.pi/2), 0, -np.sin(2*np.pi/2)],

                             [0, 1,0],

                             [np.sin(np.pi/2), 0, np.cos(np.pi/2)]])

rotate_Y_CC_135= R.from_matrix([[np.cos(3*np.pi/4), 0, -np.sin(3*np.pi/4)],

                             [0, 1,0],

                             [np.sin(-3*np.pi/4), 0, np.cos(-3*np.pi/4)]])

ngetiveY = R.from_matrix([[1,0,0,],
                               [0, 1,0],
                               [0,0,1]])

alfa = (np.pi/180)*90
rotate_Z_CC_90 = R.from_matrix([[np.cos(alfa), -np.sin(alfa),0],
                               [np.sin(np.pi/2), np.cos(alfa)  , 0],
                               [0,0,1]
                         ])

lefcameraRotation = rotate_Y_CC_45.inv()*rotate_CC_X_90

print ("rodrigues of left cam",cv.Rodrigues(lefcameraRotation.as_matrix())[0])
print ("angles of left cam ",cv.RQDecomp3x3(lefcameraRotation.as_matrix())[0] )

#Rotation matrix ( R)   from the stereocalibration 
deltaRotation=rotationMatrix_R
rightcamrotation  = deltaRotation*rotate_Y_CC_45.inv()*rotate_CC_X_90
print ("rodrigues of right cam ",cv.Rodrigues(rightcamrotation.as_matrix())[0])
angles = angles =cv.RQDecomp3x3(rightcamrotation.as_matrix())[0]
print ("angles of right cam ",angles )

#calculating the right camera rotation and translation 
rightCameraTranslation= (rotate_Y_CC_45*deltaRotation.inv()).apply(translation_calibration)+leftCameraTranslation # do the transformmation to the delta and add the offset ( left cam position )
print("distance between the two cameras in cm ", np.linalg.norm(leftCameraTranslation-rightCameraTranslation)*100)
print("distance on X axis in cm ",(rightCameraTranslation[0]-leftCameraTranslation[0])*100)
print("left camera world position ",leftCameraTranslation)
print("right camera world position ",rightCameraTranslation)


