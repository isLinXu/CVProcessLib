
import cv2
import numpy as np

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

    decision_green = np.int8(g > b) + np.int8(g > r + 30)
    decision_red = np.int8(r > b + 30) + np.int8(r > g + 30)
    decision_blue = np.int8(b > g + 30) + np.int8(b > r + 30)
    red_area = calcArea(img, decision_red)
    green_area = calcArea(img, decision_green)
    cv2.imshow('decision_green', decision_green)
    cv2.imshow('decision_red', decision_red)
    cv2.waitKey()

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

if __name__ == '__main__':
    file = '/home/linxu/Desktop/武高所图片测试/状态指示器/5.png'
    img = cv2.imread(file)
    tag = color_trace(img)
    print('tag', tag)