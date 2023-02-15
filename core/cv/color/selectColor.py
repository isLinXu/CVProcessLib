#!/usr/bin/env python3
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-05-26 上午11:54
@desc: 滚动条拖动选取颜色hsv空间
'''

from __future__ import division
import cv2
import numpy as np


def nothing(*arg):
    pass


def setTrackbar(setcolor='red'):
    # 初始化滑动条数值
    if setcolor == 'red':
        icol = (0, 100, 80, 10, 255, 255)  # Red
    elif setcolor == 'yellow':
        icol = (18, 0, 196, 36, 255, 255)  # Yellow
    elif setcolor == 'blue':
        icol = (89, 0, 0, 125, 255, 255)  # Blue
    elif setcolor == 'green':
        icol = (36, 202, 59, 71, 255, 255)  # Green
    else:
        icol = (0, 0, 0, 255, 255, 255)

    cv2.namedWindow('colorTest')
    # Lower阈值范围滑动条
    cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
    cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
    cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
    # Higher阈值范围滑动条
    cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
    cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
    cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)


def slideColor(frame):
    while True:
        # 从GUI滑块获取HSV值。
        lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
        lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
        lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
        highHue = cv2.getTrackbarPos('highHue', 'colorTest')
        highSat = cv2.getTrackbarPos('highSat', 'colorTest')
        highVal = cv2.getTrackbarPos('highVal', 'colorTest')

        # 显示原始图像。
        cv2.imshow('frame', frame)

        # 可选择不同的模糊方法
        frameBGR = cv2.GaussianBlur(frame, (7, 7), 0)
        # frameBGR = cv2.medianBlur(frameBGR, 7)
        # frameBGR = cv2.bilateralFilter(frameBGR, 15 ,75, 75)
        # kernal = np.ones((15, 15), np.float32)/255
        # frameBGR = cv2.filter2D(frameBGR, -1, kernal)
        # cv2.imshow('frameBGR_kernal', frameBGR)

        # 显示模糊图像。
        cv2.imshow('blurred', frameBGR)
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

        # 定义HSV值颜色范围
        colorLow = np.array([lowHue, lowSat, lowVal])
        colorHigh = np.array([highHue, highSat, highVal])
        mask = cv2.inRange(hsv, colorLow, colorHigh)
        # 显示mask
        cv2.imshow('mask-plain', mask)

        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

        # 显示形态变换遮罩
        cv2.imshow('mask', mask)

        # 将遮罩放在原始图像的上方。
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # 显示最终输出图像
        cv2.imshow('colorTest', result)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            print('colorLow', colorLow)
            print('colorHigh', colorHigh)
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    # file = '/home/linxu/PycharmProjects/LineCheck/image1/158.jpg'
    file = '/home/linxu/Desktop/山西焦化项目/YeWeiShi/1_900_0_1/12.jpg'
    frame = cv2.imread(file)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    setTrackbar('red')
    slideColor(frame)
