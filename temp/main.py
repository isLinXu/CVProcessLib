

import cv2
import os, sys


if __name__ == '__main__':

    print('hello world')
    # 基本路径
    RUN_PATH = os.path.dirname(__file__)
    # 图片路径
    filepath = RUN_PATH + '/images/src/HIT_20210514141956.jpeg'
    src = cv2.imread(filepath)

    # img = rsimg[1:h + 1, 1:w + 1]
    # rimg = src[500:1200,120:1000]

    # 剪切区域
    rimg = src[200:1200, 500:1400]
    cv2.imshow('src', src)
    cv2.imshow('rimg', rimg)
    # cv2.imwrite(RUN_PATH+'/images/dst/'+'rimg.png', rimg)
    # canny = cv2.Canny(src,0,255)
    hsv = cv2.cvtColor(rimg,cv2.COLOR_RGB2HSV)
    # cv2.imwrite(RUN_PATH + '/images/dst/' + 'hsv.png', hsv)
    h, s, v = cv2.split(hsv)  # 通道分离为h,s,v三个
    # cv2.imshow('hsv', hsv)
    # cv2.imshow("H", h)
    # cv2.imshow("S", s)
    # cv2.imshow("V", v)

    # cv2.imwrite(RUN_PATH + '/images/dst/' + 'H.png', h)
    # cv2.imwrite(RUN_PATH + '/images/dst/' + 'S.png', s)
    # cv2.imwrite(RUN_PATH + '/images/dst/' + 'V.png', v)

    # cv2.imshow('canny',canny)
    # gray = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
    # r,g,b = cv2.split(rimg)
    # cv2.imshow("R", r)
    # cv2.imshow("G", g)
    # cv2.imshow("B", b)
    # cv2.imwrite(RUN_PATH + '/images/dst/' + 'R.png', r)
    # cv2.imwrite(RUN_PATH + '/images/dst/' + 'G.png', g)
    # cv2.imwrite(RUN_PATH + '/images/dst/' + 'B.png', b)
    cv2.waitKey()