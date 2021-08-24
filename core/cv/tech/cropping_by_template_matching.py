'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code uses Template Matching to crop images. 
-- Status:      In progress
================================================================================
'''

import numpy as np
import cv2
import os
import sys


def matcher(waldo, puzzle):

    template = waldo
    template = cv2.cvtColor(waldo, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
    (tH, tW) = template.shape[:2]
    h, w = gray.shape[:2]
    print(h, w, tH, tW)
    found = (0, 0)
    if h < tH or w < tW:
        template = cv2.resize(template, (w, tH))
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    if found == (0, 0) or maxVal > found[0]:
        found = (maxVal, maxLoc)

    (_, maxLoc) = found
    if maxLoc != 0:
        startX, startY = (int(maxLoc[0]), int(maxLoc[1]))
        endX, endY = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))
        roi = puzzle[startY:endY, startX:endX]
        return roi
    else:
        return puzzle


fileMode = "jpg"
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        ext = filename[filename.rfind("."):].lower()
        if ext == "."+fileMode:
            fn = os.path  .join(root, filename)
            image = cv2.imread(fn)
            ho, wo = image.shape[:2]
            if wo/ho < 1:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            ho, wo = image.shape[:2]
            print(fn)
            h, w = image.shape[:2]
            waldo = cv2.imread("temp.jpg")
            roi = matcher(waldo, puzzle)
            cv2.imwrite(filename, roi)
