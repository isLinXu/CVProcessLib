# -*- coding: utf-8 -*-
#
# 点检测与识别
# Author: alex
# Created Time: 2020年03月21日 星期六 11时50分41秒
import numpy as np
from sklearn.cluster import DBSCAN


def cluster_points(point_img, eps=3, min_samples=2, metric='manhattan'):
    """交点聚类
    :param point_img 交点图像
    :param eps, min_sample, metric: dbscan聚类参数
    :return n_clusters: int: 聚类数量
    :return points: list: 每个类别的中心点
    :return labels: list: 图像上说有交点的所属类别
    """
    idx = np.argwhere(point_img == 255)
    if len(idx) < 3:
        return 0, [], []

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(idx)
    labels = db.labels_
    n_clusters = max(labels) + 1

    # 计算交点的核心点
    points = [np.average(idx[labels == i], axis=0)
              for i in range(n_clusters)]
    return n_clusters, points, labels


def point_on_line(point, a, b, e=0.01):
    """判断点是否在直线y=ax+b上，允许一定的误差
    :param point [y, x]
    :param a,b float 直线参数
    :param e float 允许的误差
    :return bool 该点是否在直线上
    """
    y, x = point
    return abs(a*x+b-y) < e
