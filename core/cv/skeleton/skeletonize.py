import cv2
from skimage import morphology
import numpy as np

def medial_axis_skeleton(img, value=125):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('binary', binary)
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8) * 255

    return dist_on_skel

def scikit_skeletonize(img, value=125):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('binary', binary)
    # cv2.imwrite("binary.png",binary)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255

    return skeleton

if __name__ == '__main__':
    # file_path = '/home/linxu/Desktop/图片2.png'
    file_path = '/home/linxu/Desktop/无人机巡检项目/输电杆塔照片素材/输电杆塔照片素材/杆塔倒塌/1.JPG'
    # file_path = '/home/linxu/Desktop/无人机巡检项目/输电杆塔照片素材/输电杆塔照片素材/杆塔倒塌/6.jpg'
    img = cv2.imread(file_path)
    img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    cv2.imshow('img', img)
    dst1 = scikit_skeletonize(img)
    cv2.imshow('skeleton', dst1)
    cv2.waitKey()
    dst2 = medial_axis_skeleton(img)
    cv2.imshow('dist_on_skel', dst2)
    cv2.waitKey()