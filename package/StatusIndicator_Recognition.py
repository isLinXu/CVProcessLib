# -*- coding: utf-8 -*-
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-06-05
@desc: 状态指示器识别相关技术
'''
import cv2
import numpy as np


# 根据阈值提取图片符合条件的像素点进行判断函数
def color_trace(img, mask):
    carea = 1.0
    carclength=1.0
    # 对原图像和创建好的掩模进行位运算
    res = cv2.bitwise_and(img, img, mask=mask)
    # 灰度化
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # 边缘提取
    canny = cv2.Canny(gray, 0, 128)
    cv2.imshow('res', res)
    # cv2.imshow('img', img)
    # cv2.imshow('gray', gray)
    # cv2.imshow('canny', canny)
    cv2.waitKey()
    # 轮廓查找
    contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # num =0

    arealist = []
    arclist = []

    for c in range(len(contours)):
        cnt = contours[c]
        M = cv2.moments(cnt)
        perimeter = cv2.arcLength(cnt, True)
        temp = 20
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        w_o, h_o, c_o = img.shape
        if (perimeter > 100):
            # cv2.drawContours(img, [cnt], 0,255,0)
            # cv2.imshow('drawmask', img)
            # cv2.waitKey()
            area = cv2.contourArea(cnt)
            # print('perimeter', perimeter)
            # print('area', area)
            arealist.append(area)

            arclist.append(perimeter)
    if arealist:
        area = arealist[int(np.argmax(arealist))]
        carclength = arclist[int(np.argmax(arclist))]
    else:
        area = 10.0
    carea = area

    return carea,carclength


def judge_open_close(img):
    # 针对武高所项目设置如下颜色阈值
    red1_lower = np.array([0, 100, 0])  # 红色下限
    red1_upper = np.array([255, 255, 255])  # 红色上限
    red2_lower = np.array([0, 130, 0])  # 红色下限
    red2_upper = np.array([10, 255, 234])  # 红色上限

    green1_lower = np.array([50, 30, 0])  # 绿色上限
    green1_upper = np.array([120, 255, 35])  # 绿色下限
    green2_lower = np.array([74, 0, 0])  # 绿色上限
    green2_upper = np.array([185, 65, 45])  # 绿色下限


    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(img_hsv, red1_lower, red1_upper)
    mask_red1 = cv2.medianBlur(mask_red1, 7)
    mask_red2 = cv2.inRange(img_hsv, red2_lower, red2_upper)
    mask_red1 = cv2.medianBlur(mask_red2, 7)
    mask_green1 = cv2.inRange(img_hsv, green1_lower, green1_upper)
    mask_green1 = cv2.medianBlur(mask_green1, 7)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2, None, None)

    # 调试图像
    # cv2.imshow('mask_red', mask_red1)
    # cv2.imshow('mask_red2', mask_red2)
    # cv2.imshow('mask_red', mask_red)
    # cv2.imshow('mask_green', mask_green1)
    # cv2.waitKey()

    # 分别计算两个色域最大的面积和弧长
    area_red,arclength_red = color_trace(img, mask_red)
    area_green,arclength_green = color_trace(img, mask_green1)

    # print('area_red', area_red, 'arclength_red',arclength_red)
    # print('area_green', area_green, 'arclength_green', arclength_green)

    # 根据色域参数进行判定内容
    resultinfo = {'max_area': 10, 'name': 'unknown', 'tag': True}
    if (area_green > area_red and arclength_green > arclength_red):
        resultinfo = {'max_area': area_green, 'name': '绿色|分|open', 'tag': True}
    else:
        resultinfo = {'max_area': area_red, 'name': '红色|合|close', 'tag': False}
    print(resultinfo)
    return resultinfo['tag']


if __name__ == "__main__":
    '''True:green(1),False:red(0)'''
    # file = '/home/linxu/Desktop/武高所图片测试/状态指示器/8.png'
    file = '/home/hxzh02/PycharmProjects/cvprocess-lib/images/状态指示器/1.jpeg'
    img = cv2.imread(file)
    cv2.imshow('img', img)
    cv2.waitKey()
    is_true = judge_open_close(img)
    # is_true = resultinfo['tag']
    # print(resultinfo)
    print(is_true)

