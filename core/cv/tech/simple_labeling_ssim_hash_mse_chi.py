'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code finds similarity (SSIM), distance (Hash), 
                mean square error (MSE) and Chi-square distance (Chi) for 
                comparison between two images descriptors. 
-- Status:      In progress
================================================================================
'''


from skimage.metrics import structural_similarity as ssim

import cv2
import os
import sys
import imutils
import pickle
import numpy as np
from imutils import paths
import os
import sys
import cv2
import pickle
import numpy as np
from PIL import Image
import imagehash

cutoff = 5


def histogram(image):
    hists = []

    for chan in cv2.split(image):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist)

    return hists


def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    return d


listed = list(paths.list_images(sys.argv[1]))

imagePaths = sorted(listed, key=lambda e: e)
imageA = np.zeros((1500, 100), dtype='uint8')
cv2.imwrite('mask.png', imageA)

nameimageA = 'mask.png'

H, W = imageA.shape[:2]
print(nameimageA)
for imagePath in imagePaths:
    filenameB = imagePath.split(os.path.sep)[-1]
    nameimageB = imagePath
    hash0 = imagehash.average_hash(Image.open(nameimageA))
    hash1 = imagehash.average_hash(Image.open(nameimageB))
    imageB = cv2.imread(nameimageB)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    imageB = cv2.resize(imageB, (W, H))
    D = chi2_distance(histogram(imageA), histogram(imageB))
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    similarity = ssim(imageA, imageB, multichannel=True)
    if .8 < similarity < 1:
        L = "D   "+filenameB+", " + \
            str(hash0 - hash1)+','+str(similarity)+','+str(err)+','+str(D)+"\n"
    elif .6 < similarity < .8:
        L = "BS  "+filenameB+", " + \
            str(hash0 - hash1)+','+str(similarity)+','+str(err)+','+str(D)+"\n"
    elif 0 < similarity < 0.6:
        L = "BR  "+filenameB+", " + \
            str(hash0 - hash1)+','+str(similarity)+','+str(err)+','+str(D)+"\n"

    print(L)
 #   fileout.writelines(L)
