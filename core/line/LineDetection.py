
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-02-24 上午11:16
@desc: 霍夫线检测
 HoughLines(image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None)
 第一个参数image：是canny边缘检测后的图像
 第二个参数rho和第三个参数theta：对应直线搜索的步长。在本例中，函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线。
 最后一个参数threshold：是经过某一点曲线的数量的阈值，超过这个阈值，就表示这个交点所代表的参数对(rho, theta)在原图像中为一条直线
 """

 """
 HoughLinesP概率霍夫变换（是加强版）使用简单，效果更好，检测图像中分段的直线（而不是贯穿整个图像的直线)
 第一个参数是需要处理的原图像，该图像必须为cannay边缘检测后的图像；
 第二和第三参数：步长为1的半径和步长为π/180的角来搜索所有可能的直线
 第四个参数是阈值，概念同霍夫变换
 第五个参数：minLineLength-线的最短长度，比这个线短的都会被忽略。
 第六个参数：maxLineGap-两条线之间的最大间隔，如果小于此值，这两条线就会被看成一条线
 """
'''

import cv2
import numpy as np

from core.tools.image_enhancement import m_s_r_c_r

def lineCenter(line):
    """求线段中心点"""
    #line = [x1,y1,x2,y2]
    px = line[0] + line[2]
    py = line[1] + line[3]
    return (px/2,py/2)

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

def nodeDistance(node1, node2):
    """点到点的直线距离"""
    distance = ((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2) ** 0.5
    return distance

def linesGetLongest(lines):
    """直线组中筛选出最长的一条直线"""
    line = lines[0]
    dd = 0
    for i in range(0, len(lines)):
        if (lineAngle(lines[i]) == 90 or lineAngle(lines[i]) == 180):
            continue
        di = nodeDistance((lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]))
        if di > dd:
            dd = di
            line = lines[i]
    return line

def line_detection(image):
    """
    直线检测求相关参数
    :param image:
    :return: lcenter,langle
    """
    w,h,c = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    edges = cv2.Canny(gray, 0, 255, apertureSize=3)     # apertureSize是sobel算子窗口大小
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # 指定步长为1的半径和步长为π/180的角来搜索所有可能的直线
    # print(len(lines))
    ntheta = []
    nrho = []
    nlines = []
    if lines is None:
        langle = 0
        return (w / 2, h / 2), langle
    if len(lines) != 0:
        for line in lines:
            # 获取极值ρ长度和θ角度
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            # 分别获取x轴值,与y轴值
            x0 = a * rho
            y0 = b * rho
            length = 1000
            # 获取这条直线最大值点x1
            x1 = int(x0 + length * (-b))
            y1 = int(y0 + length * (a))
            x2 = int(x0 - length * (-b))
            # 获取这条直线最小值点y2　　其中*1000是内部规则
            y2 = int(y0 - length * (a))
            nline = [x1, y1, x2, y2]
            # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 划线
            # cv2.imshow("image-lines", image)
            # cv2.waitKey()
            nrho.append(rho)
            ntheta.append(theta)
            nlines.append(nline)
        rline = linesGetLongest(nlines)
        langle = lineAngle(rline)
        lcenter = lineCenter(rline)
        print(langle)
        print(rline)
        return lcenter, langle
    else:
        langle = 0
        return (w/2,h/2),langle



def line_detect_possible_demo(image):  # 检测出可能的线段
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=120, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line_detect_possible_demo", image)


if __name__ == '__main__':
    path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/1/image_720.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/5/image_60.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/4/image_960.jpg'

    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/2/image_160.jpg'
    # path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/2/image_20.jpg'
    # path = '/home/linxu/Desktop/2/test.png'
    src = cv2.imread(path)
    cv2.imshow('src', src)

    src = m_s_r_c_r(src, sigma_list=[15, 80, 250])
    cv2.imshow('msrcr1', src)

    line_detection(src)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

