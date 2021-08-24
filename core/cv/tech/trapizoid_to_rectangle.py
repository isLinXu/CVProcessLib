'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code is for converting trapizoid  to rectangle.
                TRANSFRRIING 4 CURRENT  CORNERS T  RAPIZOIS BY ORDER  TL,TR,BR,BL. 
-- Status:      In progress
================================================================================
'''


import cv2
import imutils
import sys
import os
import numpy as np
saliency = cv2.saliency.StaticSaliencyFineGrained_create()


def saliantMap(image):
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    threshMap = cv2.threshold(saliencyMap, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return threshMap


def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def rectangle(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    retr, mask = cv2.threshold(gray_image, 190, 255, cv2.THRESH_BINARY)
    mask = cv2.Canny(gray_image, 50, 100)
    mask = saliantMap(image)
    mask = mask == 255
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    cropped = image[x0:x1, y0:y1]
    return warped


def trans(image):
    h, w = image.shape[:2]
    ho = h
    wo = w
    pts = np.array([(int(1*w/18), int(0)), (int(17*w/18), int(0)),
                    (int(wo), int(ho)), (int(0), int(ho))], dtype="float32")

    warped = four_point_transform(image, pts)

    return warped


fileMode = "jpg"
for root, dirs, files in os.walk(sys.argv[1]):
    for filename in files:
        ext = filename[filename.rfind("."):].lower()
        if ext == "."+fileMode:
            fn = os.path.join(root, filename)
            imagePath = fn
            image = cv2.imread(imagePath)
            warped = trans(image)
            warped = rectangle(warped)
            cv2.imwrite(filename, warped)
