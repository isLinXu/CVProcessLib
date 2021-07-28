'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-05-26 上午11:54
@desc: 颜色分割
'''

import cv2
import numpy as np
img = cv2.imread('/home/linxu/PycharmProjects/LineCheck/image1/60.jpg')
print(img.shape)
# Step1. 转换为HSV
hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 用颜色分割图像
low_range = np.array([0, 0, 180])
high_range = np.array([255, 255, 255])
th = cv2.inRange(hue_image, low_range, high_range)
cv2.imshow('hue_image', th)
cv2.waitKey()
