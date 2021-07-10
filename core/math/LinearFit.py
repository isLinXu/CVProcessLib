# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: whang@ikeeko.com
@time: 2019-05-28 上午11:23
@desc: 线性拟合技术
    点：
        node的结构为(x, y)或者[x, y]
        nodes为包含多个node的列表
    线：
        line的结构为(x1, y1, x2, y2)或者[x1, y1, x2, y2]
        lines为包含多个line的列表
    三角形：
        tria(triangle)的结构为[(x1, y1), (x2, y2), (x3, y3)]
        trias为包含多个tria的列表
    矩形：
        rect的结构为[x, y, w, h]
        rects为包含多个rect的列表
    四边形：
        quad(quadrilateral)的结构为[[[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]]]
        quads为包含多个quad的列表
    轮廓：
        contour的结构为[[[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]], [[x5, y5]], [[x6, y6]]...]
        contours为包含多个contour的列表
    圆形 ：
        circle的结构为[cnode, radius]
        circles为包含多个circle的列表

----------------------点集拟合函数------------------------
    一、RLS（最小二乘法）直线拟合：
        1.RLS（最小二乘法）直线拟合
           https://blog.csdn.net/lovetaozibaby/article/details/99482973
        2.opencv方法
            一个距离最小化函数，包含最小二乘法

    二、numpy方法
        1.polyfit:

----------------------直线组拟合函数-------------------------

    一、简单去重方法
        1.一定间隔线段组拟合（取线段最长）

----------------------矩形框组拟合函数-------------------------

    一、简单去重方法
        1.框体组包含拟合（筛选出外框体）

----------------------四边形框组拟合函数-------------------------

    一、简单去重方法
        1.一定间隔框体组拟合（取框体面积最大）

----------------------圆形框组拟合函数-------------------------

    一、简单去重方法
        1.一定间隔框体组拟合（取框体面积最大）

'''

import cv2
import numpy as np
from scipy.optimize import leastsq
from core.math.PlaneGeometry import nodeDistance, nodeAllDistance, nodeLineRelation
from core.math.Chart import chartValueRoute
from utils.ListHelper import listRemoveNpArray
from utils.FuncHelper import timereport

# ----------------------点集拟合函数-------------------------

def pointsLeastsqFit(nodes):
    """拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标"""
    # 提取x, y
    x = np.array([p[0] for p in nodes])
    y = np.array([p[1] for p in nodes])
    #  直线拟合残差计算
    def residuals(p, x, y_):
        k, b = p
        return y_ - (k * x + b)
    r = leastsq(residuals, [1, 0], args=(x, y))
    k, b = r[0]  # 最小二乘直线拟合的斜率和偏移
    # 计算这条直线任意一个点的坐标，组成直线
    y = int(100*k + b)
    # 返回拟合直线
    fitline = [0, b, 100, y]
    return fitline

def pointsOpencvFit(nodes):
    """点集opencv线性拟合"""
    # 拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标
    nodes = np.array(nodes)
    # param： 距离参数，跟所选的距离类型有关，值可以设置为0。
    # reps, aeps： 第5、6个参数用于表示拟合直线所需要的径向和角度精度，通常情况下两个值均被设定为1e-2。
    fit = cv2.fitLine(nodes, cv2.DIST_L2, 0, 0.01, 0.01)
    k = fit[1] / fit[0]
    b = fit[3] - k * fit[2]
    # 计算这条直线任意一个点的坐标，组成直线
    y = int(100*k + b)
    # 返回拟合直线
    fitline = [0, b, 100, y]
    return fitline

def pointsNumpyFit(nodes):
    """点集中通过线性拟合"""
    # 拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标
    # 提取x, y
    x = np.array([p[0] for p in nodes])
    y = np.array([p[1] for p in nodes])
    # 用一次多项式 y=kx + b 拟合这些点，fit是(a, b)
    # numpy的polyfit没办法表示跟x轴平行的拟合线，导致拟合不准bug
    fit = np.polyfit(y, x, 1)
    # 生成多项式对象 k*x+b
    fitfn = np.poly1d(fit)
    # 计算这条直线左侧0点的横坐标
    xmin = int(fitfn(0))
    # 计算这条直线右侧点的横坐标
    xmax = int(fitfn(100))
    # 返回拟合直线
    fitline = [xmin, 0, xmax, 100]
    return fitline

def pointsNodeFit(nodes, fitnode, Side):
    """多个点选择最适宜点技术"""
    n_nodes = []
    if nodes:
        #  计算两个点的最短距离 去重
        min = 10000
        rnode = nodes[0]
        for node in nodes:
            dis = nodeDistance(node, fitnode)
            if dis < min:
                min = dis
                rnode = node
        if min < Side:
            n_nodes = [rnode]
        else:
            n_nodes = [fitnode]
    return n_nodes

# ----------------------直线组拟合函数-------------------------

def linesDeduplication(lines, Side):
    """直线组中多条直线的拟合去重技术"""
    n_lines = []
    #  计算两条线段的最短距离 去重
    #  可以通过距离判断和直线相对角度来判断并把重合线段筛选为一条，选择最长的线段。
    for line in lines:
        addresult = 1
        x1, y1, x2, y2 = line[0]
        for n_line in n_lines[:]:
            # 取线段中点
            midnode = ((n_line[0] + n_line[2]) / 2, (n_line[1] + n_line[3]) / 2)
            # 算距离，垂足
            dis, footp, _ = nodeLineRelation(midnode, line[0])
            # 新线段长度
            newline_distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
            # 旧线段长度
            nowline_distance = (n_line[0] - n_line[2]) ** 2 + (n_line[1] - n_line[3]) ** 2
            if dis < Side:
                if newline_distance < nowline_distance:
                    # 舍弃新line
                    addresult = 0
                else:
                    # 舍弃旧line
                    n_lines.remove(n_line)
        if addresult == 1:
            # 添加新line
            n_lines.append([x1, y1, x2, y2])
    return n_lines

# ----------------------矩形框组拟合函数-------------------------

def rectsOverlapFilter(rects):
    """矩形框中多个矩形框重叠去重技术,取面积大的"""
    # 一般配合cv2.boundingRect使用
    n_rects = []
    for rect in rects:
        addresult = 1
        rect_x, rect_y, rect_w, rect_h = rect[:4]
        rect_s = rect_w * rect_h
        for n_rect in n_rects[:]:
            if n_rect != rect:
                nrect_x, nrect_y, nrect_w, nrect_h = n_rect[:4]
                nrect_s = nrect_w * nrect_h
                if (nrect_x + nrect_w // 2 in range(rect_x, rect_x + rect_w) and nrect_y + nrect_h // 2 in range(rect_y, rect_y + rect_h)) \
                    or (rect_x + rect_w // 2 in range(nrect_x, nrect_x + nrect_w) and rect_y + rect_h // 2 in range(nrect_y, nrect_y + nrect_h)):
                        if rect_s > nrect_s:
                            n_rects.remove(n_rect)
                        else:
                            addresult = 0
            else:
                addresult = 0
        if addresult == 1:
            # 添加新rect
            n_rects.append(rect)
    return n_rects

def rectsIncludeFilter(rects):
    """矩形框中多个矩形框包含去重技术,取外包含的"""
    # 一般配合cv2.boundingRect使用
    n_rects = []
    for rect in rects:
        addresult = 1
        rect_x, rect_y, rect_w, rect_h = rect[:4]
        for n_rect in n_rects[:]:
            if n_rect != rect:
                nrect_x, nrect_y, nrect_w, nrect_h = n_rect[:4]
                if nrect_y in range(rect_y, rect_y + rect_h) and nrect_y + nrect_h in range(rect_y, rect_y + rect_h) \
                    and nrect_x in range(rect_x, rect_x + rect_w) and nrect_x + nrect_w in range(rect_x, rect_x + rect_w):
                    n_rects.remove(n_rect)
                if rect_y in range(nrect_y, nrect_y + nrect_h) and rect_y + rect_h in range(nrect_y, nrect_y + nrect_h) \
                    and rect_x in range(nrect_x, nrect_x + nrect_w) and rect_x + rect_w in range(nrect_x, nrect_x + nrect_w):
                    addresult = 0
            else:
                addresult = 0
        if addresult == 1:
            # 添加新rect
            n_rects.append(rect)
    return n_rects

def rectsMinAreaFilter(rects, n=1, node=[], maxdistance=50):
    """矩形框中多个矩形框取面积大的前n个且最接近node的框体"""
    # 一般配合cv2.boundingRect使用
    n_rects = []
    for rect in rects:
        rect_x, rect_y, rect_w, rect_h = rect[:4]
        if node:
            distance = ((rect_x + rect_w // 2 - node[0])**2 + (rect_y + rect_h // 2 - node[1])**2)**0.5
            distancebool = distance < maxdistance
        else:
            distancebool = 1
        if distancebool:
            n_rects.append(rect)
    if n_rects:
        n_rects = sorted(n_rects, key=lambda x: x[2]*x[3], reverse=True)
        if len(n_rects) > n:
            n_rects = n_rects[:n]
    return n_rects

# ----------------------四边形框组拟合函数-------------------------

def quadsDeduplication(quads, side, mode='max'):
    """四边形框中多个四边形框体的拟合去重技术"""
    # 一般配合cv2.approxPolyDP使用
    n_quads = []
    #  计算Quads中的两条线段的最短距离 去重
    #  可以通过距离判断和直线相对角度来判断并把重合线段筛选为一条，选择最长的线段。
    for quad in quads:
        addresult = 1
        x1, y1 = quad[0][0]
        x2, y2 = quad[1][0]
        x3, y3 = quad[2][0]
        x4, y4 = quad[3][0]
        for n_quad in n_quads[:]:
            dist = []
            for node in n_quad:
                # 所有4个点距离添加集合
                n_x = node[0][0]
                n_y = node[0][1]
                length1 = ((n_x - x1)**2 + (n_y - y1)**2) ** 0.5
                length2 = ((n_x - x2)**2 + (n_y - y2)**2) ** 0.5
                length3 = ((n_x - x3)**2 + (n_y - y3)**2) ** 0.5
                length4 = ((n_x - x4)**2 + (n_y - y4)**2) ** 0.5
                dist.append([length1, length2, length3, length4])
            # 筛选4个点 对应4个点的最小距离
            mindis, pos, value = chartValueRoute(dist)
            # 点与点之间 线段总和小于阈值
            if mindis < side:
                if mode == 'max':
                    # 留面积大的那个，即无序交点的6根线（含对角线）相加，取线段最长的就是面积最大的
                    # cv2.contourArea，需要有序的交点，故不支持。
                    if nodeAllDistance(quad) < nodeAllDistance(n_quad):
                        # 舍弃新quad
                        addresult = 0
                    else:
                        # 舍弃旧quad
                        n_quads = listRemoveNpArray(n_quads, n_quad)
                elif mode == 'min':
                    # 留面积小的
                    if nodeAllDistance(quad) > nodeAllDistance(n_quad):
                        # 舍弃新quad
                        addresult = 0
                    else:
                        # 舍弃旧quad
                        n_quads = listRemoveNpArray(n_quads, n_quad)

        if addresult == 1:
            # 添加新quad
            n_quads.append(np.array([[[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]]]))
    return n_quads

# ----------------------圆形框组拟合函数-------------------------

def circlesDeduplication(circles, side, mode='max'):
    """圆形框中多个圆形框体的拟合去重技术"""
    n_circles = []
    for circle in circles:
        tag = 1
        cnode = circle[0]
        radius = circle[1]
        # 去重
        for n_circle in n_circles[:]:
            ncnode = n_circle[0]
            nradius = n_circle[1]
            if nodeDistance(cnode, ncnode) < side:
                # 替换更大圆
                if mode == 'max':
                    if radius < nradius:
                        tag = 0
                    else:
                        n_circles = listRemoveNpArray(n_circles, n_circle)
                # 替换更小圆
                elif mode == 'min':
                    if radius > nradius:
                        tag = 0
                    else:
                        n_circles = listRemoveNpArray(n_circles, n_circle)
        if tag == 1:
            n_circles.append(circle)
    return n_circles