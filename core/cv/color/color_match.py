
import cv2
import numpy as np


# 根据颜色提取对应色域
def getColorMask(img):
    '''
    根据颜色提取对应色域
    :param img:
    :return:
    # colorLow [0 0 1]
    # colorHigh [  0 255 255]
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 掩模提取色域
    lower = (0, 0 ,1)
    higher = (0,255,255)
    mask = cv2.inRange(img_hsv, lower, higher)
    # mask = cv2.medianBlur(mask, 3)
    return mask

if __name__ == '__main__':
    img_path = '/home/hxzh02/文档/plane-project_beijing/lib4mod/linebreak/output/16_4050.png'
    src = cv2.imread(img_path)
    cv2.imshow('src', src)

    mask_red = getColorMask(src)
    cv2.imshow('mask_red', mask_red)
    cv2.waitKey()
    cv2.waitKey()