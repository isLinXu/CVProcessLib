'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This codes is for detection and ecxtraction of any text 
                in wild range.
                It uses Deep Learning method.. 
-- Status:      In progress
================================================================================
'''


from imutils.object_detection import non_max_suppression
from imutils import paths
from keras.models import load_model
import numpy as np
import imutils
import cv2
import os
import sys
import time
import math
min_confidence = 0.01
listed = list(paths.list_images(sys.argv[1]))
imagePaths = sorted(listed, key=lambda e: e)

for imagePath in imagePaths:
    filename = imagePath.split(os.path.sep)[-1]
    nameimage = imagePath
    image = cv2.imread(imagePath)

    (Ho, Wo) = image.shape[:2]
    no = 1
    partition = Ho/int(no)
    orig = image
    (H, W) = image.shape[:2]
    (newW, newH) = 320, 320
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet("east.pb")

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    yha = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue
            else:

                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                yha.append(startY)
                TMP = orig[startY:endY, startX:endX]
                tname = filename.replace(
                    ".jpg", "_"+str(startX)+"_"+str(startY)+".jpg")
                tname = "TXT\\"+tname
                try:
                    cv2.imwrite(tname, (TMP))

                    print("image written")
                except:
        if len(yha) != 0:
            orig = orig[int(np.min(yha))-10:int(np.max(yha))+70]
            h, w = orig.shape[:2]
            if w > 60 and h > 20:

                cv2.imwrite("c:\\DIGIT\\"+filename, orig)
