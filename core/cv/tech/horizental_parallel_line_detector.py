'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code is for detecting Horizenal Parallel lines 
                using Hough Line detection. 
-- Status:      In progress
===============================================================================
'''


import cv2
import sys
import os
import math
import numpy as np


def applyCanny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


img = cv2.imread(sys.argv[1])

ho, wo = img.shape[:2]

img = cv2.resize(img, (int(wo*0.10), int(ho*0.10)))
ho, wo = img.shape[:2]
blue, green, red = cv2.split(img)

blue_edges = cv2.Canny(blue, 200, 250)
green_edges = cv2.Canny(green, 200, 250)
red_edges = cv2.Canny(red, 200, 250)

edges = blue_edges | green_edges | red_edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = applyCanny(img, sigma=0.33)
for i in range(2, 18, 2):
    lines = cv2.HoughLinesP(edges, 1, math.pi/i	, 2, None, 50, 1)

    for line in lines[0]:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        cv2.line(img, pt1, pt2, (0, 0, 255), 1)
        print(pt1, pt2)


cv2.imshow("linesDetected", img)
cv2.waitKey(0)
