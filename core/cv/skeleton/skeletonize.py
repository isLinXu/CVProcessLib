import cv2
from skimage import morphology
import numpy as np

def medial_axis_skeleton(img, value=125):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('binary', binary)
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8) * 255

    return dist_on_skel

def scikit_skeletonize(img, value=125):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('binary', binary)
    # cv2.imwrite("binary.png",binary)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255

    return skeleton


def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy()  # don't clobber original
    skel = img.copy()

    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:, :] = eroded[:, :]
        if cv2.countNonZero(img) == 0:
            break

    return skel


if __name__ == '__main__':
    file_path = '/home/hxzh02/文档/无人机数据集整理/杆塔图片汇总/data/img/200.jpg'
    # img = cv2.imread('/home/linxu/Desktop/无人机巡检项目/输电杆塔照片素材/输电杆塔照片素材/电线断线/DJI_0007.JPG')
    img = cv2.imread(file_path)
    # img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    cv2.imshow('img', img)
    dst1 = scikit_skeletonize(img)
    cv2.imshow('skeleton', dst1)
    cv2.waitKey()
    dst2 = medial_axis_skeleton(img)
    cv2.imshow('dist_on_skel', dst2)
    cv2.waitKey()
