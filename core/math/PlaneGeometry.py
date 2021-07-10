# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: whang@ikeeko.com
@time: 2019-05-27 下午16:10
@desc: 平面几何
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
'''

import math, cv2
import numpy as np
from core.math.Chart import chartValueRoute

# ----------------------基础相关函数-------------------------

def kToAngle(k):
    """直线斜率转化为角度"""
    # rad = math.atan(k)
    # angle = math.degrees(rad)
    angle = cv2.fastAtan2(k, 1)
    if angle > 180:
        angle = angle - 180
    if angle > 90:
        angle = angle - 180
    return angle

# ----------------------点相关函数-------------------------

def nodeRotate(node1, node2, angle):
    """node1绕node2旋转angle角度后的新坐标"""
    # 平面坐标系和笛卡尔坐标系之间的转换
    angle = angle * math.pi / 180
    # Angle为角度需转化运算，大于0时为顺时针，小于0时为逆时针
    x = (node1[0] - node2[0]) * math.cos(angle) + (node1[1] - node2[1]) * math.sin(angle) + node2[0]
    y = - (node1[0] - node2[0]) * math.sin(angle) + (node1[1] - node2[1]) * math.cos(angle) + node2[1]
    nnode = (x, y)
    return nnode

def cornerNodeSort(rect, nodes):
    """输入nodes，整理所有node，生成对应角的nodes(目前只支持矩形)"""
    nnodes = [[], [], [], []]
    x1, y1, w1, h1 = rect[:4]
    if nodes and len(nodes) >= 4:
        # 集合上的动态规划问题，最优配对问题
        dist = []
        # 找出矩框对应图像边框四个边角的位置，即各角到对应角动态分配最小距离。
        for i in range(len(nodes)):
            x, y = nodes[i][:2]
            length1 = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
            length2 = ((x - x1) ** 2 + (y - (y1 + h1)) ** 2) ** 0.5
            length3 = ((x - (x1 + w1)) ** 2 + (y - (y1 + h1)) ** 2) ** 0.5
            length4 = ((x - (x1 + w1)) ** 2 + (y - y1) ** 2) ** 0.5
            dist.append([length1, length2, length3, length4])
        # 筛选4个点 对应4个点的最小距离
        mindis, pos, value = chartValueRoute(dist)
        for i in range(len(pos)):
            nnodes[pos[i]] = nodes[i]
        # 排序按跟[0, 0], [0, 240], [320, 240], [320, 0]各点位置顺序排序
        # 根据矩形距离判断是否旋转 卡片是4:3
        dist1 = ((nnodes[0][0] - nnodes[1][0]) ** 2 + (nnodes[0][1] - nnodes[1][1]) ** 2) ** 0.5
        dist2 = ((nnodes[1][0] - nnodes[2][0]) ** 2 + (nnodes[1][1] - nnodes[2][1]) ** 2) ** 0.5
        # 比较长短，确定透视变换方式，统一为横向（320,240）进行变换，阈值需要40，才有明显长宽比
        if dist1 > dist2 + 40:
            # 确保选取短的一边先开始
            b = nnodes.pop(0)
            nnodes.append(b)
    else:
        nnodes = []
    return nnodes

# ----------------------点线相关函数-------------------------

def nodeLineGradient(node1, node2):
    """斜率"""
    x_ = node2[0] - node1[0]
    y_ = node2[1] - node1[1]
    if x_ == 0:
        x_ = 0.0000000000000001
    k = y_ / x_
    return k

def nodeLineAngle(node1, node2, tag='x+'):
    """点线对应x轴角度"""
    x_ = node2[0] - node1[0]
    y_ = node2[1] - node1[1]
    # fastAtan2函数得出的角度是以X轴正方向为0°方向，然后角度确定按照 逆时针 方向，以360°为终点，角度范围0°- 360°
    angle = cv2.fastAtan2(y_, x_)
    # 转化为对应线段按照 逆时针 方向，旋转多少度，跟X轴正方向为0°方向重合
    angle = 360 - angle
    if tag == 'y+':
        angle = angle + 90
    if tag == 'x-':
        angle = angle + 180
    if tag == 'y-':
        angle = angle + 270
    if angle > 360:
        angle = angle - 360
    return angle

def nodeDistance(node1, node2):
    """点到点的直线距离"""
    distance = ((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2) ** 0.5
    return distance

def nodeAllDistance(nodes):
    """取所有点的连接线段长度总和"""
    value = 0
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            value = value + (((nodes[i][0][0] - nodes[j][0][0])**2 + (nodes[i][0][1] - nodes[j][0][1])**2) ** 0.5)
    # 所有线段都计算多一遍 除于2
    value = value / 2
    return value

def isNodeDistanceApproach(node1, node2, node3, minlength=5):
    """判断三点是否相近，阈值为minlength"""
    distance1 = nodeDistance(node1, node2)
    distance2 = nodeDistance(node1, node3)
    distance3 = nodeDistance(node2, node3)
    if sum((distance1, distance2, distance3)) / 3 < minlength:
        return 1
    return 0

def isAllNodeDistanceApproach(nodes, minlength=5):
    """判断各点是否相近，阈值为minlength"""
    for i in range(0, len(nodes)):
        for j in range(i+1, len(nodes)):
            distance = nodeDistance(nodes[i], nodes[j])
            if distance < minlength:
                return 1
    return 0

def nodeLineRelation(node, line):
    """点到线的距离，垂足，对称点"""
    # https://blog.csdn.net/panfengzjz/article/details/80377501
    array_longi = np.array([line[2] - line[0], line[3] - line[1]])
    array_trans = np.array([line[2] - node[0], line[3] - node[1]])
    # 用向量计算点到直线距离，dot为矩阵乘法，注意转成浮点数运算
    array_temp = (float(array_trans.dot(array_longi)) / array_longi.dot(array_longi))
    array_temp = array_longi.dot(array_temp)
    distance = np.sqrt((array_trans - array_temp).dot(array_trans - array_temp))
    # 垂足
    foot = (line[2] - array_temp[0], line[3] - array_temp[1])
    # 对称点与原点的中心点为垂足
    snode = (foot[0] * 2 - node[0], foot[1] * 2 - node[1])
    # 返回 距离，垂足，对称点
    return distance, foot, snode

# ----------------------线相关函数-------------------------

def lineGradient(line):
    """斜率"""
    x_ = line[2] - line[0]
    y_ = line[3] - line[1]
    if x_ == 0:
        x_ = 0.0000000000000001
    k = y_ / x_
    return k

def lineAngle(line):
    """线对应x轴角度"""
    x_ = line[2] - line[0]
    y_ = line[3] - line[1]
    angle = cv2.fastAtan2(y_, x_)
    if angle > 180:
        angle = angle - 180
    if angle > 90:
        angle = angle - 180
    return angle

def lineCrossAngle(line1, line2):
    """两条直线夹角"""
    arr_0 = np.array([(line1[2] - line1[0]), (line1[3] - line1[1])])
    arr_1 = np.array([(line2[2] - line2[0]), (line2[3] - line2[1])])
    cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))   # 注意转成浮点数运算
    return np.arccos(cos_value) * (180/np.pi)

def lineCrossProduct(line):
    """直线内向量叉乘"""
    linelist = []
    linelist.append(line[1] - line[3])  # y1-y2
    linelist.append(line[2] - line[0])  # x2-x1
    linelist.append(line[0] * line[3] - line[2] * line[1])  # x1*y2-x2*y1
    return linelist

def lineIntersectionPoint(line1, line2):
    """两条直线相交输出交点"""
    # 线性代数 行列式求解法
    line1list = lineCrossProduct(line1)
    line2list = lineCrossProduct(line2)
    d = line1list[0] * line2list[1] - line2list[0] * line1list[1]
    # d为0的情况下，说明两直线平行。
    if d == 0:
        d = 0.0000000000000001
    x = round((line1list[1] * line2list[2] - line2list[1] * line1list[2]) * 1.0 / d, 2)
    y = round((line1list[2] * line2list[0] - line2list[2] * line1list[0]) * 1.0 / d, 2)
    # tag为交点是否在线段上（不含端点）
    if ((x - line1[0]) * (x - line1[2]) <= 0) & ((x - line2[0]) * (x - line2[2]) <= 0) & \
        ((y - line1[1]) * (y - line1[3]) <= 0) & ((y - line2[1]) * (y - line2[3]) <= 0):
        tag = 1
    else:
        tag = 0
    return tag, (x, y)

def linesGetLongest(lines):
    """直线组中筛选出最长的一条直线"""
    line = lines[0]
    dd = 0
    for i in range(0, len(lines)):
        di = nodeDistance((lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]))
        if di > dd:
            dd = di
            line = line[i]
    return line

# ----------------------三角形相关函数-------------------------

def isIRTriangle(tria, lmin=0, lmax=100):
    """判断是否为近似等腰直角三角形"""
    result = 0
    node1, node2, node3 = tria[:3]
    distance1 = nodeDistance((node1[0], node1[1]), (node2[0], node2[1]))
    distance2 = nodeDistance((node1[0], node1[1]), (node3[0], node3[1]))
    distance3 = nodeDistance((node2[0], node2[1]), (node3[0], node3[1]))
    distancelist = sorted([distance1, distance2, distance3])
    # 根据勾股定理与两条短边算出斜边长度,从而判断是不是等腰直角三角形
    if abs(distancelist[1] - distancelist[0]) < lmax*0.1 and distance1 < lmax and distance2 < lmax and distance3 < lmax:
        if abs(distancelist[1] - distancelist[0]) > lmin*0.1 and distance1 > lmin and distance2 > lmin and distance3 > lmin:
            hypotenusedistance = abs(np.sqrt(np.square(distancelist[0]) + np.square(distancelist[1])) - distancelist[2])
            if hypotenusedistance < lmax*0.15 and hypotenusedistance > lmin*0.15:
                result = 1
    return result

# ----------------------四边形相关函数-------------------------

def quadArea1(nodes):
    """计算四边形的面积"""
    # 基于向量积计算不规则四边形的面积
    # 计算面积的正负，可判断标注坐标是顺时针(负)还是逆时针(正)。
    temp_det = 0
    for idx in range(3):
        temp = np.array([nodes[idx], nodes[idx+1]])
        temp_det += np.linalg.det(temp)
    temp_det += np.linalg.det(np.array([nodes[-1], nodes[0]]))
    return abs(temp_det*0.5)

def quadArea(nodes):
    """计算多边形的面积"""
    # 基于向量积计算不规则多边形的面积, 坐标点需要按顺序（逆时针或顺时针）选取
    i_count = len(nodes)
    area_temp = 0
    for i in range(i_count):
        area_temp += nodes[i][0] * nodes[(i+1) % i_count][1] - nodes[(i+1) % i_count][0] * nodes[i][1]
    return abs(0.5 * area_temp)

# maxArea = 12000
# nodes = [[494, 116], [226, 128], [231, 283], [500, 272], [498, 228], [497, 272], [235, 285], [223, 148], [490, 114], [500, 228]]
# snodes = []
# for i1 in range(0, len(nodes)):
#     for i2 in range(i1 + 1, len(nodes)):
#         for i3 in range(i2 + 1, len(nodes)):
#             for i4 in range(i3 + 1, len(nodes)):
#                 bnodes = [nodes[i1], nodes[i2], nodes[i3], nodes[i4]]
#                 s = quadArea(bnodes)
#                 print(s)
#                 # 面积过阈值，各点能组成矩形框
#                 if quadArea1(bnodes) > maxArea and not isAllNodeDistanceApproach(bnodes, 50):
#                     maxArea = s
#                     snodes = bnodes
# print(maxArea, snodes)

# ----------------------矩形相关函数-------------------------

def isSquare(nodes, s):
    """判断是否为近似正方形"""
    # s为四边标准差阈值，其可代表形变系数，因为正方形有可能内外倾斜
    result = 0
    # 对应的四个点需要有序，无序可以先经过CornerNodeSort整理一次
    m_left = nodeDistance(nodes[0], nodes[1])
    m_right = nodeDistance(nodes[2], nodes[3])
    m_top = nodeDistance(nodes[0], nodes[3])
    m_bottom = nodeDistance(nodes[1], nodes[2])
    m_line = [m_left, m_right, m_top, m_bottom]
    # 求四边的标准差，来确定各长宽的稳定值
    std_distance = np.std(m_line, ddof=1)
    if std_distance < s:
        result = 1
    return result

def rectDistance(rect1, rect2):
    """矩形1的四个点到矩形2对应的4个点的距离"""
    x1, y1, w1, h1 = rect1[:4]
    x2, y2, w2, h2 = rect2[:4]
    distance1 = nodeDistance((x1, y1), (x2, y2))
    distance2 = nodeDistance((x1 + w1, y1), (x2 + w2, y2))
    distance3 = nodeDistance((x1, y1 + h1), (x2, y2 + h2))
    distance4 = nodeDistance((x1 + w1, y1 + h1), (x2 + w2, y2 + h2))
    alldistance = distance1 + distance2 + distance3 + distance4
    return alldistance

# ----------------------轮廓相关函数-------------------------

def contourAreaRate(contour1, contour2):
    """计算轮廓间的比率"""
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)
    if area1 == 0 or area2 == 0:
        ratio = 0
    else:
        ratio = int(area1 * 1.0 / area2) * 10
    return ratio

def contourCenter(contours):
    """计算轮廓的中心点"""
    M = cv2.moments(contours)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy

# ----------------------圆形相关函数-------------------------

def isCircle(contour):
    # 寻找最小包围圆
    cnode, radius = cv2.minEnclosingCircle(contour)
    diff = []
    for node in contour:
        node = node[0]
        dist = nodeDistance(node, cnode)
        diff.append(abs(dist - radius))
    diff = np.array(diff)
    # 临界值为3，如果有更大的圆检测，再修改函数
    if np.mean(diff) < 3:
        return 1
    return 0
