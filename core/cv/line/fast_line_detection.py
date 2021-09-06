'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code is for Fast line detection (FLD). 
-- Status:      In progress
===============================================================================
'''

import cv2
import os
import sys
import numpy as np


def FLD(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    xha = []
    yha = []
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(gray)
    line_on_imag = fld.drawSegments(image, lines)
    cv2.imshow('line_on_imag',line_on_imag)
    cv2.waitKey()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        xha.append(x1)
        yha.append(y1)
        xha.append(x2)
        yha.append(y2)

    return xha, yha, image


def corner(img):
    imagi = img
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_img, 2000, 0.01, 350)
    corners = np.int0(corners)
    xha = []
    yha = []
    for i in corners:
        x, y = i.ravel()
        xha.append(x)
        yha.append(y)
    return xha, yha


# if __name__ == '__main__':
#     # file_path = '/home/linxu/Desktop/无人机巡检项目/输电杆塔照片素材/输电杆塔照片素材/电线断线/DJI_0011.JPG'
#     file_path = '/home/linxu/Desktop/无人机巡检项目/输电杆塔照片素材/输电杆塔照片素材/杆塔倒塌/1.JPG'
#     img = cv2.imread(file_path)
#     h, w = img.shape[:2]
#     img = cv2.resize(img, (int(w*0.2), int(h*0.2)))
#     cv2.imshow('img', img)
#
#     xha, yha, img = FLD(img)
#     cv2.imshow('fld',img)
#     cv2.waitKey()
#     # cv2.imwrite('fld.png', img)
