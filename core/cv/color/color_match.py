
import cv2
import numpy as np


# 根据颜色提取对应色域
def get_find_Color(img, lower = (0,0,0), higher= (255,255,255)):
    lowHue = lower[0]
    lowSat = lower[1]
    lowVal = lower[2]
    highHue = higher[0]
    highSat = higher[1]
    highVal = higher[2]

    # 显示原始图像。
    cv2.imshow('frame', img)

    # 可选择不同的模糊方法
    frameBGR = cv2.GaussianBlur(img, (7, 7), 0)
    # frameBGR = cv2.medianBlur(frameBGR, 7)
    # frameBGR = cv2.bilateralFilter(frameBGR, 15 ,75, 75)
    # kernal = np.ones((15, 15), np.float32)/255
    # frameBGR = cv2.filter2D(frameBGR, -1, kernal)
    # cv2.imshow('frameBGR_kernal', frameBGR)

    # 显示模糊图像。
    cv2.imshow('blurred', frameBGR)
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

    # 定义HSV值颜色范围
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    # 显示mask
    cv2.imshow('mask-plain', mask)

    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    # 显示形态变换遮罩
    cv2.imshow('mask', mask)

    # 将遮罩放在原始图像的上方。
    result = cv2.bitwise_and(img, img, mask=mask)

    # 显示最终输出图像
    cv2.imshow('colorTest', result)
    cv2.waitKey()
    return mask

# 根据颜色提取对应色域
def getColorMask(img, lower = (0,0,0), higher= (255,255,255)):
    '''
    根据颜色提取对应色域
    :param img:
    :return:
    # colorLow [0 0 1]
    # colorHigh [  0 255 255]
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 掩模提取色域
    mask = cv2.inRange(img_hsv, lower, higher)
    # mask = cv2.medianBlur(mask, 3)
    # cv2.imshow('mask-plain', mask)

    # kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    # 显示形态变换遮罩
    # cv2.imshow('mask', mask)

    return mask

if __name__ == '__main__':
    # img_path = '/home/linxu/Desktop/南宁电厂项目/OCR/0_0_0_15_0_0/1000001_20220812230113_v.jpeg'
    img_path = '/home/linxu/Desktop/南宁电厂项目/OCR/0_0_0_15_0_0/1000034_20220813222452_v.jpeg'
    src = cv2.imread(img_path)
    cv2.imshow('src', src)

    lower = (0, 70, 152)
    higher = (255, 255, 255)

    # lower = (0, 70, 140)
    # higher = (255, 140, 255)

    # lower = (0,40,117)
    # higher = (255,196,255)

    # lower = (0, 76, 135)
    # higher = (255, 195, 255)

    # lower = (0, 0 ,1)
    # higher = (0,255,255)


    mask_red = getColorMask(src, lower, higher)
    cv2.imshow('mask_red', mask_red)
    cv2.waitKey()
