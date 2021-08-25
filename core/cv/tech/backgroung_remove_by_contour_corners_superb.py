'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code removes busy background of images 
                using contours and corners.
                使用轮廓和棱角,删除图像复杂的背景
-- Status:      In progress
===============================================================================
'''

import cv2
import numpy as np
import cv2
import os
import sys

def removeBackGround():
    BLUR = 21
    CANNY_THRESH_1 = 80
    CANNY_THRESH_2 = 100
    MASK_DILATE_ITER = 1
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0, 0.0, 0.0)

    nn = 1
    file_path = '/home/hxzh02/文档/defectDetect/金属锈蚀(复件)/src/13.jpg'
    img = cv2.imread(file_path)
    cv2.imshow('img', img)
    ho, wo = img.shape[:2]
    img = cv2.resize(img, (int(wo * nn), int(ho * nn)))
    ho, wo = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    cv2.imshow('Canny',edges)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    cv2.imshow('edges',edges)
    contour_info = []

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    cv2.imshow('mask',mask)
    cv2.waitKey()
    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)

    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0
    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')

    cv2.imshow('masked', masked)
    cv2.waitKey()
    image = masked
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    img = image
    blur = cv2.bilateralFilter(gray, 16, 50, 50)
    # 角点检测
    corners = cv2.goodFeaturesToTrack(blur, 40, 0.1, 10)
    corners = np.int0(corners)
    xha = []
    yha = []

    for i in corners:
        x, y = i.ravel()

        xha.append(x)
        yha.append(y)

    x0 = min(xha)
    y0 = min(yha)
    x1 = max(xha)
    y1 = max(yha)
    print(x0, x1, y0, y1)
    cropped = img[y0:y1, x0:x1]
    cv2.imshow('cropped',cropped)
    cv2.waitKey()
    # cv2.imwrite('cropped.jpg', cropped)

if __name__ == '__main__':
    removeBackGround()
