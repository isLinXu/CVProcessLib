# -*- coding:utf-8 -*-

"""
description: 对车牌图片进行校正
"""
import cv2
import imutils
import numpy as np
from imutils.perspective import four_point_transform


# 透视矫正
def perspective_transformation(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)

    # 高斯模糊
    blurred = cv2.GaussianBlur(equ, (5, 5), 0)

    # 膨胀
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Canny边缘检测
    edged = cv2.Canny(dilate, 60, 120, 3)  # 修改滞后阈值30,120来设置线段检测精度
    cv2.imshow("edged", edged)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓框
    for i in range(0, len(cnts[1])):
        x, y, w, h = cv2.boundingRect(cnts[1][i])
        rectangle = cv2.polylines(image, cnts[1][i], 1, (30, 170, 0), 2)
        # if w*h < 40000:
        #     continue
        # rectangle = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        # cv2.imshow(str(i), rectangle)
        # cv2.waitKey(0)
        # cv2.destroyWindow(str(i))

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是OpenCV2还是OpenCV3
    print (cnts[0])
    docCnt = None
    # 确保至少找到一个轮廓
    if len(cnts) > 0:
        # 按轮廓大小降序排列
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        """新添加"""
        line1 = cv2.fitLine(cnts[0], cv2.DIST_L1, 0, 1e-2, 1e-2)
        line2 = cv2.fitLine(cnts[1], cv2.DIST_L1, 0, 1e-2, 1e-2)
        line3 = cv2.fitLine(cnts[2], cv2.DIST_L1, 0, 1e-2, 1e-2)
        line4 = cv2.fitLine(cnts[3], cv2.DIST_L1, 0, 1e-2, 1e-2)
        # print line1[2]
        # cv2.line(img, line1[2], line1[3], (0,0,255), 1, cv2.LINE_AA)
        # rect = [[line1.shape[2:2],line2.shape[2:2],line3.shape[2:2],line4.shape[2:2]]]
        # dst = [[(0,0),(0,600),(600,0),(600,600)]]
        # M = cv2.getPerspectiveTransform(rect, dst)
        # paper = cv2.warpPerspective(img, M)
        # # paper = four_point_transform(img, [line1,line2,line3,line4])
        # return paper
        """新添加"""
        for c in cnts:
            hull = cv2.convexHull(c)

            # 近似轮廓
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果我们的近似轮廓有四个点，则确定找到了纸
            if len(approx) == 4:
                docCnt = approx
                break

    # 对原始图像应用四点透视变换，以获得纸张的俯视图
    print (docCnt.reshape(4, 2))
    paper = four_point_transform(img, docCnt.reshape(4, 2))
    return paper


# 逆时针旋转图像degree角度（原尺寸）
def rotate_image(src, degree):
    # 旋转中心为图像中心
    h, w = src.shape[:2]
    # 计算二维旋转的仿射变换矩阵
    RotateMatrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1)
    # 仿射变换，背景色填充为白色
    rotate = cv2.warpAffine(src, RotateMatrix, (w, h), borderValue=(255, 255, 255))
    return rotate


# 通过霍夫变换计算角度
def cal_degree(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 50, 200, 3)
    cv2.imshow("canny_img", canny_img)
    cv2.waitKey(0)

    img_copy = img.copy()

    # 通过霍夫变换检测直线
    # 第4个参数就是阈值，阈值越大，检测精度越高
    # 由于图像不同，阈值不好设定，因为阈值设定过高导致无法检测直线，阈值过低直线太多，速度很慢
    lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 170)

    sum = 0
    # 依次画出每条线段
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            print("theta:", theta, " rho:", rho)
            a = np.cos(theta)
            b = np.sin(theta)
            if a > 0.5:  # 0.92为75度,0.5为45度,超过这个角度的线段舍弃，不参与角度均值计算
                continue
            x0 = a * rho
            y0 = b * rho
            x1 = int(round(x0 + 1000 * (-b)))
            y1 = int(round(y0 + 1000 * a))
            x2 = int(round(x0 - 1000 * (-b)))
            y2 = int(round(y0 - 1000 * a))
            # 只选角度最小的作为旋转角度
            sum += theta
            cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("img_copy", img_copy)
            cv2.waitKey(0)

    # 对所有角度求平均，这样做旋转效果会更好
    average = sum / len(lines)
    angle = average / np.pi * 180 - 90  # 度数转换
    return angle


if __name__ == '__main__':
    path = "/home/hxzh02/文档/plane-project_beijing/lib4mod/towercable/output/导线抽股.png"

    print ("请输入操作:")
    a = input()
    image = cv2.imread(path)
    image = cv2.resize(image, (640, 480))
    cv2.imshow("image", image)

    if a == 1:
        # 倾斜角度校正
        degree = cal_degree(image)
        if np.fabs(degree) > 45:
            print ("拍摄角度超过45度,请重新拍摄！")
        else:
            rotate = rotate_image(image, degree)
            cv2.imshow("rotate", rotate)
            # 透视角度校正
            perspective_img = perspective_transformation(rotate)
            cv2.imshow("perspective_img", perspective_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif a == 2:
        # 透视角度校正
        perspective_img = perspective_transformation(image)
        cv2.imshow("perspective_img", perspective_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()