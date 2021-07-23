'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-07-08 上午10:54
@desc: 颜色阈值匹配技术

'''

import numpy as np
import cv2

# 设置颜色上下界
def setColorarray(color='red'):
    '''
    设置颜色上下界
    :param color:
    :return:
    '''
    # 默认颜色上下界
    lower = np.array([0, 0, 0])
    higher = np.array([255, 255, 255])
    # 设置红色上下界
    lower_red = np.array([0, 0, 0])
    higher_red = np.array([10, 255, 44])
    # lower_red = np.array([0, 127, 30])
    # higher_red = np.array([10, 255, 255])
    # 设置黄色上下界
    lower_yellow = np.array([15, 230, 230])
    higher_yellow = np.array([35, 255, 255])
    # 设置蓝色上下界
    lower_blue = np.array([85, 240, 140])
    higher_blue = np.array([100, 255, 165])
    # 设置绿色上下界
    lower_green = np.array([34, 0, 30])
    higher_green = np.array([255, 60, 35])
    # lower_green = np.array([44,0,0])
    # higher_green = np.array([98,149,32])
    if color == 'red':
        return lower_red, higher_red
    elif color == 'yellow':
        return lower_yellow, higher_yellow
    elif color == 'blue':
        return lower_blue, higher_blue
    elif color == 'green':
        return lower_green, higher_green
    else:
        return lower, higher

# 根据颜色提取对应色域
def getColorMask(img):
    '''
    根据颜色提取对应色域
    :param img:
    :return:
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色掩模提取色域
    lower_red, higher_red = setColorarray('red')
    mask_red = cv2.inRange(img_hsv, lower_red, higher_red)
    mask_red = cv2.medianBlur(mask_red, 7)
    # 黄色掩模提取色域
    lower_yellow, higher_yellow = setColorarray('yellow')
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, higher_yellow)
    mask_yellow = cv2.medianBlur(mask_yellow, 7)
    # 蓝色掩模提取色域
    lower_blue, higher_blue = setColorarray('blue')
    mask_blue = cv2.inRange(img_hsv, lower_blue, higher_blue)
    mask_blue = cv2.medianBlur(mask_blue, 7)
    # 绿色掩模提取色域
    lower_green, higher_green = setColorarray('green')
    mask_green = cv2.inRange(img_hsv, lower_green, higher_green)
    mask_green = cv2.medianBlur(mask_green, 7)
    cv2.imshow('mask_red', mask_red)
    cv2.imshow('mask_yellow', mask_yellow)
    cv2.imshow('mask_blue', mask_blue)
    cv2.imshow('mask_green', mask_green)
    cv2.waitKey()
    return mask_red, mask_yellow, mask_blue, mask_green


def findColorContour(img, mask_red, mask_yellow, mask_blue, mask_green):
    '''
    根据对应的颜色查找轮廓并用rect标记出来
    :param img:
    :param mask_red:
    :param mask_yellow:
    :param mask_blue:
    :param mask_green:
    :return:
    '''
    cnts1, hierarchy1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts2, hierarchy2 = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts3, hierarchy3 = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts4, hierarchy4 = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cnt in cnts1:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
        cv2.putText(img, 'red', (x, y - 5), font, 0.7, (0, 0, 255), 2)
    for cnt in cnts2:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
        cv2.putText(img, 'yellow', (x, y - 5), font, 0.7, (0, 255, 255), 2)

    for cnt in cnts3:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
        cv2.putText(img, 'blue', (x, y - 5), font, 0.7, (255, 0, 0), 2)

    for cnt in cnts4:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
        cv2.putText(img, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)

    for cnt in cnts3:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
        cv2.putText(img, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    file = '/home/linxu/PycharmProjects/CVProcessLib/images/呼吸器/4.jpeg'
    img = cv2.imread(file)
    mask_red, mask_yellow, mask_blue, mask_green = getColorMask(img)
    findColorContour(img, mask_red, mask_yellow, mask_blue, mask_green)
