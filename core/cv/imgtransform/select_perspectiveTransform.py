'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-07-23 上午11:54
@desc: 鼠标点选进行透视变换
'''

import cv2
import numpy as np
import time


# 选择图像上的四个点以捕获所需区域
def draw_circle(event, x, y, flags, param):
    global img
    global pointIndex
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        print('(x,y):', (x, y))
        pts[pointIndex] = (x, y)
        print(pointIndex)
        pointIndex = pointIndex + 1


def show_window():
    while True:
        # print(pts,pointIndex-1)
        cv2.imshow('img', img)

        if (pointIndex == 4):
            break

        if (cv2.waitKey(20) & 0xFF == 27):
            break


def get_persp(image, pts):
    ippts = np.float32(pts)
    Map = cv2.getPerspectiveTransform(ippts, oppts)
    warped = cv2.warpPerspective(image, Map, (AR[1], AR[0]))
    return warped


if __name__ == '__main__':
    time.sleep(1.1)
    # path = '/home/linxu/Desktop/计量中心数据集/电表流水线0/01.png'
    path = '/home/linxu/Desktop/计量中心数据集/电表流水线1/02.png'

    img = cv2.imread(path)
    # img = cv2.resize(img, (768, 576))
    # 初始化坐标点并存储输入点
    pts = [(0, 0), (0, 0), (0, 0), (0, 0)]
    pointIndex = 0
    AR = (740, 1280)
    oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', draw_circle)
    print('Top left, Top right, Bottom Right, Bottom left')
    # 显示窗口
    show_window()
    while True:
        # _, frame = cap.read()
        warped = get_persp(img, pts)
        cv2.imshow("output", warped)
        # save output file in same path
        # cv2.imwrite("output.jpg", warped)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()
