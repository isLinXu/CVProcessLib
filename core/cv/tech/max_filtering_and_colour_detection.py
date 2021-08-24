'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code is for decreasing the number of colour range 
                in an RGB image. 
                It can be used for preparing an image for segmentation. 
-- Status:      In progress
================================================================================
'''


import numpy as np
import sys
import cv2
import time

boundaries = [

    ([86, 31, 4], [220, 88, 50]),
    ([25, 146, 190], [62, 174, 250]),
    ([103, 86, 65], [145, 133, 128]),
    ([17, 15, 100], [50, 56, 200])
]


def max_rgb_filter(image):
    (B, G, R) = cv2.split(image)

    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    return cv2.merge([B, G, R])


image = cv2.imread(sys.argv[1])

h, w = image.shape[:2]
image = cv2.resize(image, (int(w/10), int(h/10)))
h, w = image.shape[:2]
filtered = max_rgb_filter(image)
orig = image
image = filtered
cv2.imshow("filter.jpg", filtered)
cv2.waitKey(0)

for (lower, upper) in boundaries:
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(orig, orig, mask=mask)
    cv2.imshow("output", output)
    cv2.waitKey(0)
