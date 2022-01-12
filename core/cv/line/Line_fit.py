# -*- coding: utf-8 -*-

import cv2
import numpy as np

# 两个回调函数
def HoughLinesP(minLineLength):
    global minLINELENGTH
    minLINELENGTH = minLineLength + 1
    print("minLINELENGTH:", minLineLength + 1)
    tempIamge = scr.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, minLINELENGTH, 0)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(tempIamge, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imshow(window_name, tempIamge)

if __name__ == '__main__':

    # 临时变量
    minLineLength = 20

    # 全局变量
    minLINELENGTH = 20
    max_value = 100
    window_name = "HoughLines Demo"
    trackbar_value = "minLineLength"

    path = '/home/hxzh02/文档/plane-project_beijing/lib4mod/towercable/output/导线抽股.png'
    # 读入图片，模式为灰度图，创建窗口
    scr = cv2.imread(path)
    gray = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    cv2.namedWindow(window_name)

    # 创建滑动条
    cv2.createTrackbar(trackbar_value, window_name,
                       minLineLength, max_value, HoughLinesP)

    # 初始化
    HoughLinesP(20)

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
