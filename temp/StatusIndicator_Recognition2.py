# -*- coding: utf-8 -*-
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-06-05
@desc: 状态指示器识别相关技术
'''
import cv2
import numpy as np

from core.panelAbstract import panelAbstract
from core.grabcut import grabCut


def color_trace(img):
    # 颜色分割区域
    b = img[:, :, 0].astype("int16")  # blue
    g = img[:, :, 1].astype("int16")  # green
    r = img[:, :, 2].astype("int16")  # red

    decision_green = np.int8(g > b + 30) + np.int8(g > r + 30)
    decision_red = np.int8(r > b + 30) + np.int8(r > g + 30)
    decision_blue = np.int8(b > g + 30) + np.int8(b > r + 30)
    red_area = calcArea(img, decision_red)
    green_area = calcArea(img, decision_green)

    resultinfo = {'score': 0, 'name': '', 'is_close': None}
    print('green_area', green_area, 'red_area', red_area)
    area = 0
    if green_area > red_area:
        name = '绿'
        area = green_area
        is_open = True
        return is_open,area
    else:
        name = '红'
        area = red_area
        is_open = False
        return is_open,area

def judge_open_close(img):
    """状态指示器识别 open:1 close:0"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    w, h, c = img
    # 使用下采样加上采样使图片失真达到去噪效果
    img_down = cv2.pyrDown(gray, (w // 2, h // 2))
    img_up = cv2.pyrUp(img_down, (w, h))
    cv2.imshow('img_up',img_up)
    cv2.waitKey()


def calcArea(img,decision):
    w_o,h_o,c_o = img.shape
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[decision == 2] = 255
    cv2.imshow('mask', mask)
    cv2.waitKey()

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)==0:
    #     # 建议hsv取值范围
    #     red1_lower = np.array([150, 43, 46])  # 红色下限
    #     red1_upper = np.array([180, 255, 255])  # 红色上限
    #     red2_lower = np.array([0, 43, 46])  # 红色下限
    #     red2_upper = np.array([10, 255, 255])  # 红色上限
    #     green_lower = np.array([50, 20, 10])
    #     green_upper = np.array([100, 255, 255])
    #     lower = [red1_lower, red2_lower, green_lower]
    #     upper = [red1_upper, red2_upper, green_upper]
    #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # RGB图像转HSV图像
    #     # 创建掩膜，提取在三个模板阈值范围之间的像素设为255，其它为0
    #     mask0 = cv2.inRange(img_hsv, lower[0], upper[0])
    #     mask1 = cv2.inRange(img_hsv, lower[1], upper[1])
    #     mask2 = cv2.inRange(img_hsv, lower[2], upper[2])
    #     # 三个模板进行或运算
    #     mask_temp = cv2.bitwise_or(mask0, mask1, None, None)
    #     mask = cv2.bitwise_or(mask_temp, mask2, None, None)
    #
    #     # 对原图像和创建好的掩模进行位运算
    #     res = cv2.bitwise_and(img, img, mask=mask)
    #     # 灰度化
    #     gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #     # 边缘提取
    #     canny = prewitt(gray)
    #     cv2.imshow('canny', canny)
    #     cv2.waitKey()
    #
    #     mask += canny
    #     cv2.imshow('mask1', mask)
    #     cv2.waitKey()
    #     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    areas = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        points.append([x, y, w, h])
        areas.append(w * h)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)

    if areas:
        areas = np.array(areas)
        points = np.array(points)
        area = areas[np.argmax(areas)]
        point = points[np.argmax(areas)]
        x, y, w, h = point
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imshow('img', img)
        cv2.waitKey()
    else:
        area = 100.00

    rate = float(area/(w_o*h_o))
    print('rate',rate, area,(w_o*h_o))
    #　针对大区域面积特征,排除背景红色干扰,
    # if rate > 0.4:
    #     area = areas[1]

    # point = points[np.argmax(areas)]
    # point = points[0]
    # x, y, w, h = point
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    return area

from scipy import signal
# 边缘提取函数
def prewitt(I, _boundary='symm', ):
    # prewitt算子是可分离的。 根据卷积运算的结合律，分两次小卷积核运算

    # 算子分为两部分，这是对第一部分操作
    # 1: 垂直方向上的均值平滑
    ones_y = np.array([[1], [1], [1]], np.float32)
    i_conv_pre_x = signal.convolve2d(I, ones_y, mode='same', boundary=_boundary)
    # 2: 水平方向上的差分
    diff_x = np.array([[1, 0, -1]], np.float32)
    i_conv_pre_x = signal.convolve2d(i_conv_pre_x, diff_x, mode='same', boundary=_boundary)

    # 算子分为两部分，这是对第二部分操作
    # 1: 水平方向上的均值平滑
    ones_x = np.array([[1, 1, 1]], np.float32)
    i_conv_pre_y = signal.convolve2d(I, ones_x, mode='same', boundary=_boundary)
    # 2: 垂直方向上的差分
    diff_y = np.array([[1], [0], [-1]], np.float32)
    i_conv_pre_y = signal.convolve2d(i_conv_pre_y, diff_y, mode='same', boundary=_boundary)

    # 取绝对值，分别得到水平方向和垂直方向的边缘强度
    abs_i_conv_pre_x = np.abs(i_conv_pre_x)
    abs_i_conv_pre_y = np.abs(i_conv_pre_y)

    # 水平方向和垂直方向上的边缘强度的灰度级显示
    edge_x = abs_i_conv_pre_x.copy()
    edge_y = abs_i_conv_pre_y.copy()

    # 将大于255的值截断为255
    edge_x[edge_x > 255] = 255
    edge_y[edge_y > 255] = 255

    # 数据类型转换
    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)

    # 显示
    # cv2.imshow('edge_x', edge_x)
    # cv2.imshow('edge_y', edge_y)

    # 利用abs_i_conv_pre_x 和 abs_i_conv_pre_y 求最终的边缘强度
    # 求边缘强度有多重方法, 这里使用的是插值法
    edge = 0.5 * abs_i_conv_pre_x + 0.5 * abs_i_conv_pre_y

    # 边缘强度灰度级显示
    edge[edge > 255] = 255
    edge = edge.astype(np.uint8)

    return edge


# 根据轮廓判断函数
def judge_lunkuo(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 边缘提取
    edge = prewitt(gray)
    # 二值化
    _, bin = cv2.threshold(edge, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 查找轮廓
    conts, hier = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(conts):
        cnt = contour
        M = cv2.moments(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # 周长判断
        if perimeter < 500 and perimeter > 100:
            # 面积判断
            if M['m00'] > 500 and M['m00'] < 4000:
                # if  extent>=0.6 :
                # 建立掩膜，
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                pixelpoints = np.transpose(np.nonzero(mask))
                # print('pixelpoints', pixelpoints)
                judge_result = judge_color(image, pixelpoints)
                if judge_result == True:
                    return True
    return False


# 根据轮廓区域内像素点判断函数
def judge_color(img, xys):
    num = 0
    green_lower = np.array([50, 20, 10])
    green_upper = np.array([100, 255, 255])
    # 变为hsv空间
    hsvs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 筛选轮廓内在阈值范围内的像素点数
    for i in xys:
        hsv = hsvs[i[0], i[1]]
        if green_lower[0] <= hsv[0] <= green_upper[0]:
            if green_lower[1] <= hsv[1] <= green_upper[1]:
                if green_lower[2] <= hsv[2] <= green_upper[2]:
                    num += 1
    if num > 100:
        # print(num)
        return True
    else:
        return False


def method(src):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # img_hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    # 二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 15)
    # cv2.imshow('hsv', img_hsv)
    # cv2.imshow('gray', gray)
    cv2.imshow('binary', binary)
    # 取出原图的行,列,维度，辅助轮廓条件筛选
    h_o, w_o = img.shape[:2]
    # 使用下采样加上采样使图片失真达到去噪效果
    # img_down = cv2.pyrDown(binary, (w_o // 2, h_o // 2))
    # img_up = cv2.pyrUp(img_down, (w_o, h_o))
    # cv2.imshow('img_up', img_up)

    # 轮廓检测
    canny = cv2.Canny(binary, 0, 255)
    cv2.imshow('canny', canny)

    # 膨胀
    kernel = np.ones((1, 1), np.uint8)
    dilation = cv2.dilate(canny, kernel, iterations=1)
    cv2.imshow('dilation', dilation)
    cv2.waitKey()

    # 进行轮廓的查找
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    s = w_o * h_o
    # 轮廓筛选
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        center_x = w_o // 2
        center_y = h_o // 2
        approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True) * 0.2, True)

        if x in range(center_x // 1000, center_x) and y in range(center_y // 1000, center_y):
            cv2.drawContours(src, contours, i, (0, 0, 255))
            cv2.imshow('draw', src)
            cv2.waitKey()

        # if (cv2.contourArea(contours[i]) > 0.03 * s and w < w_o and h < h_o):
        #     # print('arclength', cv2.arcLength(approx, True), (w + h))
        #     # print('area', cv2.contourArea(contours[i]), 0.03 * s, 0.08 * s, s)
        #      cv2.drawContours(src, contours, i, (0, 0, 255))
        #     cv2.imshow('draw', src)
        #     cv2.waitKey()

if __name__ == '__main__':
    file = '/home/linxu/Desktop/武高所图片测试/状态指示器/5.png'
    # file = '/home/linxu/Desktop/1_0_0_1020_1_0/状态指示器/Green/9.jpeg'
    img = cv2.imread(file)
    cv2.imshow('img', img)
    # src = panelAbstract(img)
    # cv2.imshow('src', src)
    # src = grabCut(src)
    # method(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    img_mean = cv2.blur(binary, (3, 3))

    cv2.imshow('binary', binary)
    # cv2.imshow('img_mean', img_mean)

    canny = cv2.Canny(binary,100,255)

    # 边缘提取
    # canny = prewitt(binary)
    cv2.imshow('canny', canny)
    cv2.waitKey()

    # 轮廓查找
    contours,hierachy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    for i in range(len(contours)):
        contour = contours[i]
        temp = 600.00
        arclength = cv2.arcLength(contour, True)
        carea = cv2.contourArea(contour)
        # 建立掩膜
        # mask = np.zeros(binary.shape, np.uint8)
        # cv2.drawContours(mask, [contour], 0, 255, -1)
        # cv2.imshow('mask', mask)
        # print('carea', carea, 'arclength', arclength)
        # print('rate=', carea / arclength)
        cv2.drawContours(img, contours, i, (0, 0, 255))
        cv2.imshow('draw', img)
        cv2.waitKey()
        # # 周长面积筛选
        # if carea > temp and arclength < 8000:
        #     # 周长和面积比值筛选
        #     if float(carea / arclength)>3.5:
        #         # 建立掩膜
        #         mask = np.zeros(binary.shape, np.uint8)
        #         cv2.drawContours(mask, [contour], 0, 255, -1)
        #         cv2.imshow('mask', mask)
        #         print('carea', carea, 'arclength', arclength)
        #         print('rate=', carea / arclength)
        #         cv2.drawContours(src, contours, i, (0, 0, 255))
        #         cv2.imshow('draw', src)
        #         cv2.waitKey()
        #         rect = cv2.boundingRect(contour)
        #         x,y,w,h = rect
        #         print(rect)
        #         print('w/h', w/h,'h/w' , h/w)
        #         if (w / h < 1 or h / w < 1.5):
        #             ROI = src[y:y + h, x:x + w]
        #             cv2.imshow('ROI', ROI)
        #             cv2.waitKey()
        #             print('状态指示器ROI区域为', rect)
        #             cv2.imshow('mask', mask)
        #             cv2.waitKey()
