import cv2
from skimage import morphology
import numpy as np

def medial_axis_skeleton(img, value=125):
    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('binary', binary)
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8) * 255
    cv2.imshow('dist_on_skel', dist_on_skel)
    cv2.waitKey()
    return dist_on_skel

def scikit_skeletonize(img, value=125):
    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('binary', binary)
    # cv2.imwrite("binary.png",binary)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    cv2.imshow('skeleton', skeleton)
    cv2.waitKey()
    return skeleton

if __name__ == '__main__':
    img = cv2.imread('/home/hxzh02/PycharmProjects/cvprocess-lib/images/呼吸器/2.jpeg')
    dst1 = scikit_skeletonize(img)
    dst2 = medial_axis_skeleton(img)