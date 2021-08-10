# -*- coding: utf-8 -*-
#
# 图像相似性
# Author: alex
# Created Time: 2020年01月06日 星期一 16时19分23秒
import cv2


def orb_similary(img1, img2, distance_thr=0.75, **kwargs):
    """
    计算图像相似性
    :param img1 cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    :param img2 cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    :param distance_thr 匹配的距离阈值
    :param **kwargs cv2.OBR_create函数的参数
    :return similary float
    说明:
    ORB_create([, nfeatures[, scaleFactor[, nlevels[, edgeThreshold[, firstLevel[, WTA_K[, scoreType[, patchSize[, fastThreshold]]]]]]]]])
    """
    # 读取图片
    # 初始化ORB检测器
    orb = cv2.ORB_create(**kwargs)
    _, des1 = orb.detectAndCompute(img1, None)
    _, des2 = orb.detectAndCompute(img2, None)

    # 提取并计算特征点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # knn筛选结果
    matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
    # print(matches)

    # 查看最大匹配点数目
    good = [m for (m, n) in matches if m.distance < distance_thr * n.distance]
    # print(len(good))
    # print(len(matches))
    similary = float(len(good))/len(matches)
    # print("(ORB算法)两张图片相似度为:%s" % similary)
    return similary
