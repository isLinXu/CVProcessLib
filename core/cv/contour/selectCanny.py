'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-07-01 上午11:30
@desc: 滚动条拖动控制阈值检测Canny轮廓
'''
import cv2
import numpy as np
import time

# 默认初始值
lowThreshold = 0
max_lowThreshold = 100

maxThreshold = 100
max_maxThreshold = 200
kernel_size = 3

def canny_low_threshold(intial):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blur, intial,maxThreshold)  # x是最小阈值,y是最大阈值
    cv2.imshow('canny', canny)

def canny_max_threshold(intial):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blur, lowThreshold,intial)  # x是最小阈值,y是最大阈值
    cv2.imshow('canny', canny)

if __name__ == '__main__':
    img = cv2.imread('/home/linxu/PycharmProjects/CVProcess/images/表计/test1.png')
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.namedWindow('canny', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('Min threshold', 'canny', lowThreshold, max_lowThreshold, canny_low_threshold)
    cv2.createTrackbar('Max threshold', 'canny', maxThreshold, max_maxThreshold, canny_max_threshold)
    canny_low_threshold(0)

    if cv2.waitKey(0) == 27:    # 27是ESC键值
        cv2.destroyAllWindows()
