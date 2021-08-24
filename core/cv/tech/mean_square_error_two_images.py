'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code finds mean square error (MSE) 
                for comparison between two images. 
-- Status:      In progress
================================================================================
'''
import numpy as np
import cv2
import os
import sys


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


imageA = cv2.imread(sys.argv[1])
imageB = cv2.imread(sys.argv[2])
print(mse(imageA, imageB))
