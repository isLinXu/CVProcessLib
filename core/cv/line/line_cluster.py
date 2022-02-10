# -*- coding: utf-8 -*-
#
# 线段聚类
# Author: alex
# Created Time: 2020年03月26日 星期四 15时18分41秒
import numpy as np
from sklearn.cluster import DBSCAN
# from ibbd_algo.optics import Optics


def line_cluster(lines, line_types, enpoints, eps=2, min_samples=2):
    """将有交点的线段聚合在一起
    线段方程类型为:
        True: y=ax+b
        False: x=ay+b
    :param lines list 线段方程参数[(a, b)]
    :param line_types list 线段方程的类型，跟lines参数对应，取值True or False
    :param enpoints list 线段的端点，注意每个线段有两个端点: [[(y1, x1), (y2, x2)]]
    :param eps, min_samples: DBSCAN聚类参数
    :return labels
    """
    data = [(l_type, a, b, x1, y1, x2, y2)
            for ((a, b), l_type, ((x1, y1), (x2, y2))) in
            zip(lines, line_types, enpoints)]
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=distance).fit(data)
    return db.labels_
    # optics = Optics(max_radius, min_samples, distance=distance)
    # optics.fit(data)
    # return optics.cluster(cluster_thr)


def distance(line1, line2):
    """计算两个线段的距离
    :param line1,line2: [a, b, x1, y1, x2, y2]
    """
    l_type1, a1, b1, x11, y11, x12, y12 = line1
    l_type2, a2, b2, x21, y21, x22, y22 = line2

    def format_fraction(val):
        c_val = 0.000000001
        if -c_val < val < 0:
            val = -c_val
        elif c_val > val >= 0:
            val = c_val

        return val

    # 计算直线交点
    if l_type1 and l_type2:
        x0 = (b2-b1) / format_fraction(a1-a2)
        y0 = a1*x0 + b1
    elif not l_type1 and not l_type2:
        y0 = (b2-b1) / format_fraction(a1-a2)
        x0 = a1*y0 + b1
    elif l_type1 and not l_type2:
        # y=a1*x+b1 and x=a2*y+b2
        y0 = (a1*b2+b1) / format_fraction(1-a1*a2)
        x0 = a2*y0 + b2
    elif not l_type1 and l_type2:
        # x=a1*y+b1 and y=a2*x+b2
        x0 = (a1*b2+b1) / format_fraction(1-a1*a2)
        y0 = a2*x0 + b2

    def point_line_dist(x1, y1, x2, y2):
        """计算点到线的距离"""
        if x1 <= x0 <= x2 and y1 <= y0 <= y2:
            dist = 0
        else:
            # 到两端点的最小距离
            dist = min(np.linalg.norm([x0-x1, y0-y1]),
                       np.linalg.norm([x0-x2, y0-y2]))

        return dist

    # 计算到线1的距离
    dist1 = point_line_dist(x11, y11, x12, y12)
    # 计算到线2的距离
    dist2 = point_line_dist(x21, y21, x22, y22)
    return dist1 + dist2


if __name__ == '__main__':
    pass
