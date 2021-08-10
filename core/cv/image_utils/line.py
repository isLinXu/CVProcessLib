# -*- coding: utf-8 -*-
#
# 直线检测与识别
# Author: alex
# Created Time: 2020年03月20日 星期五 16时41分48秒
import cv2
import numpy as np
from math import atan
from sklearn.cluster import DBSCAN


def intersection_line(line_a, line_b):
    """计算两个线段的重叠部分"""
    a1, b1 = line_a
    a2, b2 = line_b
    if b1 <= a2 or b2 <= a1:
        return 0
    return min(b1, b2) - max(a1, a2)


def in_line_rate(line, container_line):
    """一个线段和另一个线段的重合部分，占该线段总长的占比"""
    inter = intersection_line(line, container_line)
    return inter / (line[1] - line[0])


def iou_line(line_a, line_b):
    """两个线段的重叠占比"""
    inter = intersection_line(line_a, line_b)
    if inter == 0:
        return 0
    a1, b1 = line_a
    a2, b2 = line_b
    return inter / (max(b1, b2) - min(a1, a2))


def intersection_points(line_img1, line_img2):
    """计算两个直线的交点图像"""
    points = cv2.bitwise_and(line_img1, line_img2)
    return points


def detect_lines_angle(gray, scale, line_type='row'):
    """检测直线的倾斜角度
    :param gray 灰度图
    :param scale 检测参数
    :param line_type 直线类型，值为row（横线）或者col（竖线）
    :return 倾斜弧度，如果需要转换为角度: math.degrees
    """
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, -10)
    if line_type == 'row':
        line_img = detect_row_line(binary, scale)
    elif line_type == 'col':
        line_img = detect_col_line(binary, scale)
    else:
        raise Exception('error line_type value')

    n, lines, _ = cluster_fit_lines(line_img)
    if n < 3:
        return None
    angles = [line[0] for line in lines]
    return atan(sum(angles)/len(angles))


def detect_line(gray, scale):
    """检测横线和竖线"""
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, -10)
    row_img = detect_row_line(binary, scale)
    col_img = detect_col_line(binary, scale)
    return row_img, col_img


def detect_col_line(binary, scale):
    """检测竖线"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cv2.dilate(eroded, kernel, iterations=1)


def detect_row_line(binary, scale):
    """检测横线"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cv2.dilate(eroded, kernel, iterations=1)


def cluster_lines(data, eps=3, min_samples=2, metric='manhattan'):
    """线条聚类"""
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(data)
    labels = db.labels_
    n_clusters_ = max(labels) + 1
    return n_clusters_, labels


def fit_line(points, exchange_xy=False):
    """拟合直线
    注意：如果是针对竖线，应该对xy轴进行交换，避免出现x=b这样的直线
    :params points [[y, x]] 图像坐标：[y, x]
    :param exchange_xy bool 拟合直线时，决定是否需要交换x和y轴
    :return [a, b] 直线参数：y=ax+b
    """
    X = points[:, 1]
    Y = points[:, 0]
    if exchange_xy:
        X, Y = Y, X

    line = np.polyfit(X, Y, 1)
    return line


def cluster_fit_lines(line_img, exchange_xy=False, sorted_b=False):
    """直线聚合并拟合直线
    注意：如果是竖线，类似y=b这种，应该讲x轴和y轴进行交换
    :param line_img 直线的黑白图像
    :param exchange_xy bool 拟合直线时，决定是否需要交换x和y轴
    :param sorted_b bool 是否按截距进行排序
    :return n int 直线数量
    :return lines [[a, b]] 直线方程的参数
    :return endpoints [[y, x]] 线段的端点
    """
    points_idx = np.argwhere(line_img == 255)
    if len(points_idx) < 3:
        return 0, [], []

    n, labels = cluster_lines(points_idx)
    lines = []
    endpoints = []
    for i in range(n):
        line_points_idx = points_idx[labels == i]
        line = fit_line(line_points_idx, exchange_xy=exchange_xy)
        lines.append(line)
        endpoint = get_endpoint(line, line_points_idx, exchange_xy=exchange_xy)
        endpoints.append(endpoint)

    if sorted_b and len(lines) > 1:
        data = [(l, p) for l, p in zip(lines, endpoints)]
        data = sorted(data, key=lambda x: x[0][-1])
        lines = [l for l, _ in data]
        endpoints = [p for _, p in data]

    return n, lines, endpoints


def get_endpoint(line, points_idx, exchange_xy=False):
    """获取线段的端点
    :param line  list 直线方程的参数，值为[a, b]
    :param points_idx list 图像直线上所有点的坐标，坐标格式是[y, x]
    :param exchange_xy bool 是否对x轴和y轴进行交换，默认为False，即y=ax+b，若为True, 则是x=ay+b
    :return endpoint1, endpoint2: 每个顶点格式[y, x]
    """
    a, b = line
    # 数据中点的坐标格式是[y, x]
    if exchange_xy:
        # 这个是竖线，需要找y的最大最小值
        X = [y for y, _ in points_idx]
        v_min, v_max = min(X), max(X)
        return (v_min, v_min*a+b), (v_max, v_max*a+b)

    # 对于横线，需要找到x的最大最小值
    X = [x for _, x in points_idx]
    v_min, v_max = min(X), max(X)
    return (v_min*a+b, v_min), (v_max*a+b, v_max)


def check_segment_collinear(seg1, seg2, exchange_xy=False, b_err=4.,
                            a_err=0.1):
    """判断两个线段是否共线
    :param seg1 list 线段1, 格式：[a, b, (y1, x1), (y2,  x2)]。线段所在直线y=ax+b， (y1, x1)与(y2, x2)是线段两个端点
    :param seg2 list 线段2, 格式：[a, b, (y1, x1), (y2,  x2)]
    :param exchange_xy bool 默认为False即可，如果是竖线，则设置该值为True，这时对应的直线方程是x=ay+b
    :param b_err float 直线方程中b参数允许的误差
    :param a_err float 直线方程中a参数允许的误差
    :return bool 两个线段是否共线
    """
    a1, b1 = seg1[:2]
    a2, b2 = seg2[:2]
    if abs(a1 - a2) > a_err or abs(b1 - b2) > b_err:
        return False

    # 避免两个线段有交叉
    if exchange_xy:
        # 竖线
        (v11, _), (v12, _) = seg1[2:]
        (v21, _), (v22, _) = seg2[2:]
    else:
        (_, v11), (_, v12) = seg1[2:]
        (_, v21), (_, v22) = seg2[2:]

    min1, max1 = min(v11, v12), max(v11, v12)
    min2, max2 = min(v21, v22), max(v21, v22)
    return max1 < min2 or min1 > max2


if __name__ == '__main__':
    assert intersection_line((1, 10), (20, 30)) == 0
    assert intersection_line((1, 10), (5, 30)) == 5
    assert in_line_rate((1, 10), (5, 30)) == 5/9
    assert in_line_rate((5, 30), (1, 10)) == 5/25
    assert iou_line((5, 30), (1, 10)) == 5/29
