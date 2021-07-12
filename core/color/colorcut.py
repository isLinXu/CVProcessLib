'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-05-26 上午11:54
@desc: 颜色分割
'''

import cv2
import numpy as np

# Step1. 转换为HSV
img = cv2.imread('/home/linxu/PycharmProjects/LineCheck/image/60.jpg')
print(img.shape)
img = cimg = img[360:720, 0:1280]
# black = np.zeros((int(img.shape[0] / 2), int(img.shape[1])), dtype=np.uint8)
# cimg = img[0:360, 0:1280]
# cv2.imshow('cimg', cimg)
# cv2.imshow('black', black)

# res=cv2.add(img1, img2)
# black = np.zeros((int(img.shape[0] / 2), int(img.shape[1])), dtype=np.uint8)

# cv2.copyTo(img, black)

hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Step2. 用颜色分割图像
low_range = np.array([0, 0, 180])
high_range = np.array([255, 255, 255])
th = cv2.inRange(hue_image, low_range, high_range)
cv2.imshow('hue_image', th)

# res=cv2.add(black, th)
# res = th + black
# cv2.imshow('add', res)
cv2.waitKey()

# Step3. 形态学运算，膨胀
dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

# Step4. Hough Circle
circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=10, maxRadius=20)

# Step5. 绘制
if circles is not None:
    x, y, radius = circles[0][0]
    center = (x, y)
    cv2.circle(img, center, radius, (0, 255, 0), 2)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
