'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code is for Template Matching for 
                various size of template (subimage). 
-- Status:      In progress
===============================================================================
'''

import numpy as np
import argparse
import imutils
import glob
import cv2
import sys
import os

template = cv2.imread(sys.argv[2])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

fileMode = 'jpg'
for root, dirs, files in os.walk(sys.argv[1]):

    for filename in files:

        ext = filename[filename.rfind("."):].lower()
        if ext == "."+fileMode:

            fn = os.path.join(root, filename)
            image = cv2.imread(os.path.join(root, filename))
            asl = image
            fn = os.path.join(root, filename)
            image = cv2.imread(fn)

            ho, wo = image.shape[:2]

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            found = None

            for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                resized = imutils.resize(
                    gray, width=int(gray.shape[1] * scale))
                r = gray.shape[1] / float(resized.shape[1])

                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break

                edged = cv2.Canny(resized, 50, 200)
                result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

                (_, maxLoc, r) = found
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (
                    int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

                cv2.rectangle(image, (startX, startY),
                              (endX, endY), (0, 0, 255), 2)

                image = image[startY:endY, startX:endX]
                cv2.imwrite(filename, image)
