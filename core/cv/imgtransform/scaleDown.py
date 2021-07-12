# -*- coding: utf-8 -*-
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2019-06-04 上午11：16
@desc: 图像变换技术

----------------------图像缩放技术------------------------
    类型1.等比例缩放

----------------------图像扩展技术------------------------
    类型1.先比例缩放，再居中扩展
    类型2.先圆圈截取，再居中扩展
'''

import cv2, math
import numpy as np
# ----------------------图像缩放技术------------------------


def ImgScaleDown(Image, rsize):
    """图像等比例缩放"""
    nh, nw = rsize
    # 获取图像的尺寸，先按比例缩放到dsize的大小，假设dsize = （28， 28）
    h, w = Image.shape[:2]
    sh = h / nh
    sw = w / nw
    if sh > sw:
        # 以sh缩放
        rh = max(int(h / sh), 1)
        rw = max(int(w / sh), 1)
        Image = cv2.resize(Image, (rw, rh), interpolation=cv2.INTER_CUBIC)
    else:
        # 以sw缩放
        rh = max(int(h / sw), 1)
        rw = max(int(w / sw), 1)
        Image = cv2.resize(Image, (rw, rh), interpolation=cv2.INTER_CUBIC)
    return Image, (rh, rw)

# ----------------------图像扩展技术------------------------

def ImgCenterAdd(Image, rsize, dsize, value=(255, 255, 255)):
    """图像居中扩展"""
    nh, nw = dsize
    # 先等比例缩放
    Image, (rh, rw) = ImgScaleDown(Image, rsize)
    """
    Various border types, image boundaries are denoted with '|'
    *BORDER_REPLICATE: aaaaaa | abcdefgh | hhhhhhh
    *BORDER_REFLECT: fedcba | abcdefgh | hgfedcb
    *BORDER_REFLECT_101: gfedcb | abcdefgh | gfedcba
    *BORDER_WRAP: cdefgh | abcdefgh | abcdefg
    *BORDER_CONSTANT: iiiiii | abcdefgh | iiiiiii
    with some specified 'i'
    """
    # 横向拓展，余数需要调整
    aw = (nw - rw) // 2
    if (nw - rw) % 2 == 0:
        leftw = aw
    else:
        leftw = aw + 1

    # 竖向拓展，余数需要调整
    ah = (nh - rh) // 2
    if (nh - rh) % 2 == 0:
        toph = ah
    else:
        toph = ah + 1

    Image = cv2.copyMakeBorder(Image, toph, ah, leftw, aw, cv2.BORDER_CONSTANT, value=value)
    return Image

def ImgCircleAdd(Image, radius):
    """图像圆圈截取，居中扩展"""
    # 判断Image
    h, w = Image.shape[:2]
    # 生成内显示模板
    circleIn = Image.copy()
    # 统一白色
    circleIn[:] = 255
    if min(h, w) // 2 < radius:
        radius = min(h, w) // 2
    circleIn = cv2.circle(circleIn, (w // 2, h // 2), int(radius), (0, 0, 0), -1)
    # 图像或操作
    Image = cv2.bitwise_or(circleIn, Image)
    return Image
