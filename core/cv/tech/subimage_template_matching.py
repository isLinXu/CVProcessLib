'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code is for template matching of an image. 
-- Status:      In progress
================================================================================
'''
import cv2
import sys
import os
import numpy as np
import imutils


def TemplateMatching(image, template):

    puzzle = image
    waldo = template
    waldoHeight, waldoWidth = waldo.shape[:2]
    result = cv2.matchTemplate(puzzle, waldo, cv2.TM_CCOEFF)
    (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

    topLeft = maxLoc
    botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
    roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

    mask = np.zeros(puzzle.shape, dtype="uint8")
    cv2.imshow("Puzzle", puzzle)
    cv2.imshow("Waldo", waldo)
    cv2.imshow("roi", roi)
    cv2.waitKey(0)


image = cv2.imread(sys.argv[1], 0)
(h, w) = image.shape[:2]
image = cv2.resize(image, (int(w/10), int(h/10)))
(h, w) = image.shape[:2]
template = cv2.imread(sys.argv[2], 0)
TemplateMatching(image, template)
