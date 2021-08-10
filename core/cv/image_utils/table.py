"""
表格聚类: 将有关联的线段聚合在一起
Author: alex
Created Time: 2020年05月25日
"""
import numpy as np
from sklearn.cluster import DBSCAN

# 线段a，b参数的最大值
A_MAX = 1e8
B_MAX = 1e8
# 斜率最小值
A_MIN = 1e-8
# 距离最大值
D_MAX = 1e8


def table_lines_cluster(lines, eps=3, min_samples=2):
    """表格线段聚类
    将有关联的线段聚合在一起
    线段：y = a*x+b，(x1, y1)和(x2, y2)是其两个端点，两种表示形式
    1. 用线段的两个端点来表示一个线段：(x1, y1, x2, y2)
    2. 如果已经计算出参数a和b，则线段：(x1, y1, x2, y2, a, b)
    :params lines list 线段列表
    :params eps, min_samples: DBSCAN聚类所使用的参数
    :return labels np.array 聚类结果
    """
    new_lines = []
    if len(lines[0]) == 6:
        for x1, y1, x2, y2, a, b in lines:
            a = max(min(a, A_MAX), -A_MAX)
            b = max(min(b, B_MAX), -B_MAX)
            if abs(a) < A_MIN:
                a = A_MIN

            new_lines.append([x1, y1, x2, y2, a, b])
    elif len(lines[0]) == 4:
        for x1, y1, x2, y2 in lines:
            a, b = cal_line_params(x1, y1, x2, y2)
            new_lines.append([x1, y1, x2, y2, a, b])
    else:
        raise Exception('lines: param error!')

    cluster = DBSCAN(eps=eps, min_samples=min_samples,
                     metric=distance).fit(new_lines)
    return cluster.labels_


def cal_line_params(x1, y1, x2, y2):
    """计算直线的参数"""
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    a = max(min(a, A_MAX), -A_MAX)
    b = max(min(b, B_MAX), -B_MAX)
    if abs(a) < A_MIN:
        a = A_MIN

    return a, b


def distance(line1, line2):
    """计算两个线段的距离
    a, b: 线段直线参数: y=ax+b
    x1, y1, x2, y2: 线段的两个端点
    :param line1, line2: [x1, y1, x2, y2, a, b]
    :return float 两个线段的距离
    """
    a1, b1 = line1[4:]
    a2, b2 = line2[4:]

    # 计算交点
    if abs(a1-a2) < 0.01:
        return D_MAX
    x0 = (b2-b1)/(a1-a2)
    y0 = a1 * x0 + b1

    def point_line_dist(x1, y1, x2, y2):
        """计算点到线的距离"""
        v1 = [x1-x0, y1-y0]
        v2 = [x2-x0, y2-y0]
        return min(np.linalg.norm(v1), np.linalg.norm(v2))

    # 计算到线1的距离
    dist1 = point_line_dist(*line1[:4])
    # 计算到线2的距离
    dist2 = point_line_dist(*line2[:4])
    # print(dist1+dist2)
    return dist1 + dist2


if __name__ == '__main__':
    def create_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        a = (y2-y1)/(x2-x1)
        b = y1 - a*x1
        return (x1, y1, x2, y2, a, b)

    points1 = [(1, 9), (2, 1), (7, 1.5), (10, 10)]
    points2 = [(4, 2), (5, 2), (6, 4), (3, 5)]
    data = [create_line(points1[i], points1[i+1])
            for i in range(len(points1)-1)]
    for i in range(len(points2)-1):
        data.append(create_line(points2[i], points2[i+1]))

    data.append(create_line(points2[0], points2[len(points2)-1]))
    labels = table_lines_cluster(data, eps=2.5)
    print(labels)

    data = [points1[i]+points1[i+1] for i in range(len(points1)-1)]
    for i in range(len(points2)-1):
        data.append(points2[i]+points2[i+1])

    data.append(points2[0]+points2[len(points2)-1])
    labels = table_lines_cluster(data, eps=2.5)
    print(labels)
