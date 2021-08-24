'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code is for GrabCut using mask. 
                GrabCut is an image segmentation method based on graph cuts.
-- Status:      In progress
===============================================================================
'''

import numpy as np
import sys
import time
import cv2
import os

saliency = cv2.saliency.StaticSaliencyFineGrained_create()


def saliantMap(image):
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    threshMap = cv2.threshold(saliencyMap, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return threshMap


image = cv2.imread(sys.argv[1])

h, w = image.shape[:2]
image = cv2.resize(image, (int(w/10), int(h/10)))
h, w = image.shape[:2]
mask = saliantMap(image)
roughOutput = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Rough Output", roughOutput)
cv2.waitKey(0)

mask[mask > 0] = cv2.GC_PR_FGD
mask[mask == 0] = cv2.GC_BGD

fgModel = np.zeros((1, 65), dtype="float")
bgModel = np.zeros((1, 65), dtype="float")

start = time.time()
(mask, bgModel, fgModel) = cv2.grabCut(image, mask, None, bgModel,
                                       fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)
end = time.time()
print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))

values = (
    ("Definite Background", cv2.GC_BGD),
    ("Probable Background", cv2.GC_PR_BGD),
    ("Definite Foreground", cv2.GC_FGD),
    ("Probable Foreground", cv2.GC_PR_FGD),
)

for (name, value) in values:
    print("[INFO] showing mask for '{}'".format(name))
    valueMask = (mask == value).astype("uint8") * 255

    cv2.imshow(name, valueMask)
    cv2.waitKey(0)

outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                      0, 1)
outputMask = (outputMask * 255).astype("uint8")

output = cv2.bitwise_and(image, image, mask=outputMask)

cv2.imshow("Input", image)
cv2.imshow("GrabCut Mask", outputMask)
cv2.imshow("GrabCut Output", output)
cv2.waitKey(0)
