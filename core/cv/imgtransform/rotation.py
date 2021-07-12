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

----------------------图像旋转技术------------------------
    一般图像的旋转是以图像的中心为原点，旋转一定的角度，也就是将图像上的所有像素都旋转一个相同的角度。
    类型1.旋转图像，把转出显示区域的图像截去：ImgRotate
    类型2.旋转图像，扩大图像范围来显示所有的图像：ImgAddRotate（默认扩展填充白色）
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

# ----------------------图像旋转技术------------------------

def ImgRotate(Image, NodeList, Angle):
    """图像有界旋转"""
    # 结点格式
    # NewNodeList = {'开始': [], '结束': [], 'P1': [], 'P2': []}

    # 获取图像的尺寸，然后确定中心点
    h, w = Image.shape[:2]
    # 将图像中心设为旋转中心
    cX, cY = (w // 2, h // 2)

    # 抓住旋转矩阵(应用角度的负值顺时针旋转)，然后抓住正弦和余弦
    # (即，矩阵的旋转分量)
    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    M = cv2.getRotationMatrix2D((cX, cY), -Angle, 1.0)

    # 执行实际的旋转
    # 第三个参数是输出图像的尺寸中心
    Rotated = cv2.warpAffine(Image, M, (w, h), borderValue=(255, 255, 255))

    # 返回旋转后的图像, 及旋转后的坐标
    NewNodeList = NodeRotate(NodeList, Angle, h, w, 0, 0)

    # 返回旋转后的图像, 及旋转后的坐标
    return Rotated, NewNodeList

def ImgAddRotate(Image, NewNodeDict, Angle):
    """图像无界旋转"""
    # 结点格式
    # NewNodeDict = {'开始': [], '结束': [], 'P1': [], 'P2': []}

    # 旋转angle角度，缺失背景白色（255, 255, 255）填充
    # grab the dimensions of the image and then determine the center
    # 获取图像的尺寸，然后确定中心点
    h, w = Image.shape[:2]
    # 将图像中心设为旋转中心
    cX, cY = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # angle位置参数为角度参数正值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -Angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    Rotated = cv2.warpAffine(Image, M, (nW, nH), borderValue=(255, 255, 255))

    # 返回旋转后的图像, 及旋转后的坐标
    sX = (nW - w) // 2
    sY = (nH - h) // 2
    NewNodeDict = NodeRotate(NewNodeDict, Angle, h, w, sX, sY)

    return Rotated, NewNodeDict

def NodeRotate(NodeList, Angle, h, w, sX, sY):
    """图像点旋转"""
    NewNodeDict = {}
    # 转化角度值
    Angle = Angle * math.pi / 180
    for i in NodeList:
        NewNodeDict[i] = []
        for Node in NodeList[i]:
            # 计算原版旋转节点坐标
            X = Node[0] * math.cos(Angle) - Node[1] * math.sin(Angle) - 0.5 * w * math.cos(Angle) + 0.5 * h * math.sin(
                Angle) + 0.5 * w
            Y = Node[1] * math.cos(Angle) + Node[0] * math.sin(Angle) - 0.5 * w * math.sin(Angle) - 0.5 * h * math.cos(
                Angle) + 0.5 * h
            # 位置偏移算法
            X = X + sX
            Y = Y + sY
            NewNode = (round(X), round(Y))
            NewNodeDict[i].append(NewNode)
    return NewNodeDict

def img_resize(img, size):
    # 强化黑白图中黑线条
    _, threshold_img = cv2.threshold(
        cv2.cvtColor(
            cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)
