# -*- coding: utf-8 -*-
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-06-05
@desc: 旋转矫正相关技术
'''
import cv2
import numpy as np

from core.line.LineDetection import line_detection
from core.tools.image_enhancement import m_s_r_c_r


def imgTransmat(img, nodes):
    """透视变换"""
    src = np.float32([nodes[0], nodes[1], nodes[2], nodes[3]])
    h = max(nodes[2][0][1],nodes[3][0][1]) - min(nodes[0][0][1],nodes[1][0][1])
    w = max(nodes[2][0][0],nodes[3][0][0]) - min(nodes[0][0][0],nodes[1][0][0])

    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    transform = cv2.getPerspectiveTransform(src, dst)
    transmat = cv2.warpPerspective(img, transform, (w, h))
    return transmat


def nodeSort(rect):
    """角点排序"""
    print(rect[0],rect[1],rect[2],rect[3])
    pcenter = ((rect[0][0][0]+rect[1][0][0]+rect[2][0][0]+rect[3][0][0])/4,(rect[0][0][1] + rect[1][0][1] + rect[2][0][1] + rect[3][0][1])/4)
    # print(pcenter)
    p1=(0,0)
    p2=(0,0)
    p3=(0,0)
    p4=(0,0)
    for i in range(0, len(rect)):
        print(rect[i])
        rt = rect[i]
        if rect[i][0][0] < pcenter[0] and rect[i][0][1] < pcenter[1]:
            p1 = rt
        elif rect[i][0][0] < pcenter[0] and rect[i][0][1] > pcenter[1]:
            p3 = rt
        elif rect[i][0][0] > pcenter[0] and rect[i][0][1] < pcenter[1]:
            p2 = rt
        elif rect[i][0][0] > pcenter[0] and rect[i][0][1] > pcenter[1]:
            p4 = rt
    points = [p1,p2,p3,p4]
    print(points)
    return points

def perspectiveCorrection(img):
    """透视变换矫正图像抠显区域"""
    h, w, c = img.shape
    s = w*h
    # 图像预处理
    dilation = imagePreprocessing(img)
    # 进行轮廓的查找
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    # 轮廓筛选
    for i in range(len(contours)):
        approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True)*0.1, True)
        if len(approx) == 4 and abs(cv2.contourArea(approx)) > 0.03*s and cv2.isContourConvex(approx) and abs(cv2.contourArea(approx)) < 0.15 * s:
            print('arclength', cv2.arcLength(approx, True), (w + h))
            print('area', cv2.contourArea(approx), 0.03 * s, 0.08 * s, s)
            if (cv2.arcLength(approx,True) > 0.3*(w+h) and cv2.arcLength(approx,True) < 0.9*(w+h)):
                quads.append(approx)
                cv2.drawContours(img, contours, i, (0, 0, 255))
                # cv2.imshow('draw', img)
                # cv2.waitKey()
                # print(approx)

    rect = []
    if len(quads)!= 0:
        rect = quads[0]
    print(rect)
    if len(rect)!= 0:
        points = nodeSort(rect)
        mat = imgTransmat(img,points)
    return mat


def imagePreprocessing(img):
    """图像预处理"""
    h, w, c = img.shape
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('gray', gray)

    # 二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)
    # cv2.imshow('binary', binary)

    # 使用下采样加上采样使图片失真达到去噪效果
    img_down = cv2.pyrDown(binary, (w // 2, h // 2))
    img_up = cv2.pyrUp(img_down, (w, h))

    # 轮廓检测
    canny = cv2.Canny(img_up, 0, 128)
    # cv2.imshow('canny', canny)

    # 膨胀
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(canny, kernel, iterations=1)
    # cv2.imshow('dilation',dilation)
    return dilation



def rotateCorrection(img):
    '''旋转矫正'''
    h, w, c = img.shape
    s = w*h
    # 图像预处理
    dilation = imagePreprocessing(img)

    # 进行轮廓的查找
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    # 轮廓筛选
    for i in range(len(contours)):
        approx = cv2.approxPolyDP(contours[i], cv2.arcLength(contours[i], True)*0.1, True)
        if len(approx) == 4 and abs(cv2.contourArea(approx)) > 0.03*s and cv2.isContourConvex(approx) and abs(cv2.contourArea(approx)) < 0.15 * s:
            # print('arclength', cv2.arcLength(approx, True), (w + h))
            # print('area', cv2.contourArea(approx), 0.03 * s, 0.08 * s, s)
            if (cv2.arcLength(approx,True) > 0.3*(w+h) and cv2.arcLength(approx,True) < 0.9*(w+h)):
                quads.append(approx)
                # cv2.drawContours(img, contours, i, (0, 0, 255))
                # cv2.imshow('draw', img)
                # cv2.waitKey()
                # print(approx)

    rect = []
    if len(quads)!= 0:
        rect = quads[0]
    print(rect)

    if len(rect)!= 0:
        boxs = cv2.minAreaRect(rect)
        print(boxs)
        # (中心点坐标),(宽高),(angle)
        point = boxs[0]
        # 角度范围[-90，0]
        angle = boxs[2]
        # 角度范围判断
        if angle > -89 and angle <= -45:
            angle += 90
        elif angle > -45 and angle < -1:
            angle = -angle
        elif angle > -1 or angle < 1:
            angle = 0
        elif angle > 45:
            angle -= 90
        # print('angle', angle)
        # print('point', point)
        M = cv2.getRotationMatrix2D(point,angle,1)
        dst = cv2.warpAffine(img, M, (w, h))
        return dst
    else:
        lcener, langle = line_detection(img)
        M = cv2.getRotationMatrix2D(lcener, langle, 1)
        dst = cv2.warpAffine(img, M, (w, h))
        return dst

# 灰度世界算法
def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    avg_b = np.average(nimg[0])
    avg_g = np.average(nimg[1])
    avg_r = np.average(nimg[2])

    avg = (avg_b + avg_g + avg_r) / 3

    nimg[0] = np.minimum(nimg[0] * (avg / avg_b), 255)
    nimg[1] = np.minimum(nimg[1] * (avg / avg_g), 255)
    nimg[2] = np.minimum(nimg[2] * (avg / avg_r), 255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)


if __name__ == '__main__':
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/1/image_720.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/1/image_220.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/1/image_260.jpg'

    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/2/image_160.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/2/image_260.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/2/image_180.jpg'

    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/3/image_860.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/3/image_920.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/3/image_900.jpg'

    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/4/image_960.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/4/image_880.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/4/image_860.jpg'

    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/5/image_580.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/5/image_560.jpg'
    path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/5/image_540.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/5/image_280.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/2/image_20.jpg'

    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/2/image_160.jpg'
    # path = '/home/linxu/Desktop/2/test.png'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/4/image_40.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/3/image_20.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/4/image_860.jpg'
    img = cv2.imread(path)
    cv2.imshow('img', img)

    #　多尺度视网膜与颜色恢复
    msrcr1 = m_s_r_c_r(img, sigma_list=[15, 80, 250])
    # cv2.imshow('msrcr1', msrcr1)
    # cv2.imwrite('/home/linxu/Desktop/2/msrcr1.png', msrcr1)

    # 直方图均衡化
    # balance_img = his_equl_color(msrcr1)
    # cv2.imshow('balance_img', balance_img)

    dst = rotateCorrection(msrcr1)
    cv2.imshow('dst', dst)
    # cv2.imwrite('/home/linxu/Documents/Work/sources/textDetect_project/cut_out/dst2.png', dst)
    cv2.waitKey()