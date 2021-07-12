'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-07-08 上午10:54
@desc: 颜色阈值匹配技术

'''

import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
lower_red = np.array([0, 0, 0])  # 红色阈值下界
higher_red = np.array([10, 255, 44])  # 红色阈值上界
# lower_red = np.array([0, 127, 30])  # 红色阈值下界
# higher_red = np.array([10, 255, 255])  # 红色阈值上界


lower_yellow = np.array([15, 230, 230])  # 黄色阈值下界
higher_yellow = np.array([35, 255, 255])  # 黄色阈值上界
lower_blue = np.array([85,240,140])
higher_blue = np.array([100,255,165])
lower_green = np.array([34,0,30])
higher_green = np.array([255,60,35])
# lower_green = np.array([44,0,0])
# higher_green = np.array([98,149,32])

file = '/home/linxu/Desktop/武高所图片测试/状态指示器/1.png'
# file = '/home/linxu/Desktop/武高所实地相关材料/wugaosuo/1_0_0_1020_1_0/1.jpeg'
frame=cv2.imread(file)
img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_red = cv2.inRange(img_hsv, lower_red, higher_red)  # 可以认为是过滤出红色部分，获得红色的掩膜
mask_yellow = cv2.inRange(img_hsv, lower_yellow, higher_yellow)  # 获得绿色部分掩膜
mask_yellow = cv2.medianBlur(mask_yellow, 7)  # 中值滤波
mask_red = cv2.medianBlur(mask_red, 7)  # 中值滤波
mask_blue = cv2.inRange(img_hsv, lower_blue, higher_blue)  # 获得绿色部分掩膜
mask_blue = cv2.medianBlur(mask_blue, 7)  # 中值滤波
mask_green = cv2.inRange(img_hsv, lower_green, higher_green)  # 获得绿色部分掩膜
mask_green = cv2.medianBlur(mask_green, 7)  # 中值滤波



#mask = cv2.bitwise_or(mask_green, mask_red)  # 三部分掩膜进行按位或运算
print(mask_red)
cv2.imshow('mask_red', mask_red)
cv2.imshow('mask_green', mask_green)
cv2.waitKey()
cnts1, hierarchy1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓检测 #红色
cnts2, hierarchy2 = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓检测 #红色
cnts3, hierarchy3 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in cnts1:
    (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
    cv2.putText(frame, 'red', (x, y - 5), font, 0.7, (0, 0, 255), 2)
for cnt in cnts2:
    (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
    cv2.putText(frame, 'blue', (x, y - 5), font, 0.7, (0, 0, 255), 2)

for cnt in cnts3:
    (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
    cv2.putText(frame, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
cv2.imshow('frame', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()