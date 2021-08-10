# -*- coding: utf-8 -*-
#
# box相关函数
#
# Author: alex
# Created Time: 2020年03月10日 星期二
import numpy as np


def intersection_area(box1, box2):
    """计算两个矩形的重叠面积"""
    x1, y1, xb1, yb1 = box1
    x2, y2, xb2, yb2 = box2

    # 相交矩形
    ax, ay, bx, by = max(x1, x2), max(y1, y2), min(xb1, xb2), min(yb1, yb2)
    if ax >= bx or ay >= by:
        return 0

    # 重叠面积
    in_area = (bx-ax) * (by-ay)
    # print((ax, ay, bx, by))
    # print('相交面积：%d' % in_area)
    return in_area


def iou(box1, box2):
    """计算两个矩形的交并比"""
    in_area = intersection_area(box1, box2)
    if in_area == 0:
        return 0.

    x1, y1, xb1, yb1 = box1
    x2, y2, xb2, yb2 = box2
    area1 = abs((xb1-x1) * (yb1-y1))
    area2 = abs((xb2-x2) * (yb2-y2))
    return in_area / (area1 + area2 - in_area)


def in_box_rate(box, container_box):
    """判断一个box在一个容器box里的占比
    :return float 例如返回值为0.6，则表示box在容器box中的面积占box的60%
    """
    in_area = intersection_area(box, container_box)
    if in_area == 0:
        return 0.
    x1, y1, xb1, yb1 = box
    area = abs((xb1-x1) * (yb1-y1))
    return in_area / area


def boxes_in_row(box1, box2):
    """判断两个box是否在同一行"""
    if iou(box1, box2) > 0:
        return False     # 不能有交集
    if box1[0] > box2[0]:
        box1, box2 = box2, box1

    _, y1, xb1, yb1 = box1
    x2, y2, _, yb2 = box2
    if xb1 > x2:
        return False    # box2必须在box1的右边

    # 垂直方向上交集
    min_yb, max_y = min(yb1, yb2), max(y1, y2)
    if min_yb <= max_y:
        return False    # 如果没有交集
    max_h = max(yb1, yb2) - min(y1, y2)
    min_h = min_yb - max_y

    # 高度差
    h1, h2 = yb1-y1, yb2-y2
    h1, h2 = min(h1, h2), max(h1, h2)
    print((h2-h1)/h1)

    # 水平方向需要相邻 & 重叠部分超过80% & 高度差不能超过20%
    # TODO 这里的参数可能不是最优的，可以经过测试调整
    return min_h/max_h > 0.8 and (x2-xb1) < 2*min_h and (h2-h1)/h1 < 0.2


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)
    :param box 四个顶点坐标[x1, y1, x2, y2, x3, y3, x4, y4]
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1+x3+x2+x4)/4.0
    cy = (y1+y3+y4+y2)/4.0
    w = (np.sqrt((x2-x1)**2+(y2-y1)**2)+np.sqrt((x3-x4)**2+(y3-y4)**2))/2
    h = (np.sqrt((x2-x3)**2+(y2-y3)**2)+np.sqrt((x1-x4)**2+(y1-y4)**2))/2
    # x = cx-w/2
    # y = cy-h/2
    # sinA = ((y2+y3)/2 - (y1+y4)/2) / w
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def rotate_cut_img(img, box, degree, wh, center,
                   rotate=False, leftAdjust=1.0, rightAdjust=1.0):
    """四边形旋转并裁剪图像，通常和solve搭配使用
    :param img PIL图像
    :param box 四个顶点坐标[x1, y1, x2, y2, x3, y3, x4, y4]
    :param degree 选择角度, 对应solve函数中的angle
    :param wh 对应solve函数中的w和h
    :param center 中心点坐标, 对应solve函数中的cx和cy
    """
    # 原图坐标
    # degree, w, h, x_center, y_center = solve(box)
    w, h = wh
    x_center, y_center = center
    xmin_ = min(box[0::2])
    xmax_ = max(box[0::2])
    ymin_ = min(box[1::2])
    ymax_ = max(box[1::2])

    # 第一次裁剪
    img = img.crop([xmin_, ymin_, xmax_, ymax_])

    # 裁剪后的中心点
    x_center = x_center - xmin_
    y_center = y_center - ymin_

    # 旋转时长度不变: 左上右下点坐标
    xmin = max(0, x_center-w/2-leftAdjust*h)
    ymin = y_center-h/2
    xmax = min(x_center+w/2+rightAdjust*h, img.size[0]-1)
    ymax = y_center+h/2

    # 按照裁剪后的中心点旋转并裁剪
    degree_ = degree*180.0/np.pi
    if rotate is False:
        crop_img = img.crop([xmin, ymin, xmax, ymax])
    else:
        if abs(degree_) <= 0.0001:
            # 不需要进行旋转
            degree_ = 0
            crop_img = img.crop([xmin, ymin, xmax, ymax])
        else:
            crop_img = img.rotate(degree_, center=(x_center, y_center))\
                .crop([xmin, ymin, xmax, ymax])

    return crop_img


if __name__ == '__main__':
    box1 = [325.8022766113281, 393.0766296386719,
            592.3567504882812, 435.80364990234375]
    box2 = [620.7103881835938, 397.5979309082031,
            660.7562255859375, 433.3531188964844]
    print(boxes_in_row(box1, box2))
    box1 = [339.4958190917969, 222.9739532470703,
            581.1708374023438, 261.8145751953125]
    box2 = [634.3968505859375, 222.9739532470703,
            670.3604125976562, 264.6916809082031]
    print(boxes_in_row(box1, box2))
