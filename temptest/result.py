import cv2
import numpy as np
from scipy import signal


# 边缘提取函数
def prewitt(I, _boundary='symm', ):
    # prewitt算子是可分离的。 根据卷积运算的结合律，分两次小卷积核运算

    # 算子分为两部分，这是对第一部分操作
    # 1: 垂直方向上的均值平滑
    ones_y = np.array([[1], [1], [1]], np.float32)
    i_conv_pre_x = signal.convolve2d(I, ones_y, mode='same', boundary=_boundary)
    # 2: 水平方向上的差分
    diff_x = np.array([[1, 0, -1]], np.float32)
    i_conv_pre_x = signal.convolve2d(i_conv_pre_x, diff_x, mode='same', boundary=_boundary)

    # 算子分为两部分，这是对第二部分操作
    # 1: 水平方向上的均值平滑
    ones_x = np.array([[1, 1, 1]], np.float32)
    i_conv_pre_y = signal.convolve2d(I, ones_x, mode='same', boundary=_boundary)
    # 2: 垂直方向上的差分
    diff_y = np.array([[1], [0], [-1]], np.float32)
    i_conv_pre_y = signal.convolve2d(i_conv_pre_y, diff_y, mode='same', boundary=_boundary)

    # 取绝对值，分别得到水平方向和垂直方向的边缘强度
    abs_i_conv_pre_x = np.abs(i_conv_pre_x)
    abs_i_conv_pre_y = np.abs(i_conv_pre_y)

    # 水平方向和垂直方向上的边缘强度的灰度级显示
    edge_x = abs_i_conv_pre_x.copy()
    edge_y = abs_i_conv_pre_y.copy()

    # 将大于255的值截断为255
    edge_x[edge_x > 255] = 255
    edge_y[edge_y > 255] = 255

    # 数据类型转换
    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)

    # 显示
    # cv2.imshow('edge_x', edge_x)
    # cv2.imshow('edge_y', edge_y)

    # 利用abs_i_conv_pre_x 和 abs_i_conv_pre_y 求最终的边缘强度
    # 求边缘强度有多重方法, 这里使用的是插值法
    edge = 0.5 * abs_i_conv_pre_x + 0.5 * abs_i_conv_pre_y

    # 边缘强度灰度级显示
    edge[edge > 255] = 255
    edge = edge.astype(np.uint8)

    return edge


# 根据轮廓判断函数
def judge_lunkuo(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 边缘提取
    edge = prewitt(gray)
    # 二值化
    _, bin = cv2.threshold(edge, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 查找轮廓
    conts, hier = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(conts):
        cnt = contour
        M = cv2.moments(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # 周长判断
        if perimeter < 500 and perimeter > 100:
            # 面积判断
            if M['m00'] > 500 and M['m00'] < 4000:
                # if  extent>=0.6 :
                # 建立掩膜，
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                pixelpoints = np.transpose(np.nonzero(mask))
                # print('pixelpoints', pixelpoints)
                judge_result = judge_color(image, pixelpoints)
                if judge_result == True:
                    return True
    return False


# 根据轮廓区域内像素点判断函数
def judge_color(img, xys):
    num = 0
    green_lower = np.array([50, 20, 10])
    green_upper = np.array([100, 255, 255])
    # 变为hsv空间
    hsvs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 筛选轮廓内在阈值范围内的像素点数
    for i in xys:
        hsv = hsvs[i[0], i[1]]
        if green_lower[0] <= hsv[0] <= green_upper[0]:
            if green_lower[1] <= hsv[1] <= green_upper[1]:
                if green_lower[2] <= hsv[2] <= green_upper[2]:
                    num += 1
    if num > 100:
        # print(num)
        return True
    else:
        return False


# 根据阈值提取图片符合条件的像素点进行判断函数
def color_trace(lower, upper, img):
    # mask_dilate = cv2.dilate(img, (9, 9), iterations = 2)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # RGB图像转HSV图像
    # 创建掩膜，提取在三个模板阈值范围之间的像素设为255，其它为0
    mask0 = cv2.inRange(img_hsv, lower[0], upper[0])
    mask1 = cv2.inRange(img_hsv, lower[1], upper[1])
    mask2 = cv2.inRange(img_hsv, lower[2], upper[2])
    # 三个模板进行或运算
    mask_temp = cv2.bitwise_or(mask0, mask1, None, None)
    mask = cv2.bitwise_or(mask_temp, mask2, None, None)

    # 对原图像和创建好的掩模进行位运算
    res = cv2.bitwise_and(img, img, mask=mask)
    # 灰度化
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # 边缘提取
    canny = prewitt(gray)
    # cv2.imshow('canny', canny)
    # cv2.waitKey()
    # 轮廓查找
    cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # num =0
    for c in cnts:
        M = cv2.moments(c)
        perimeter = cv2.arcLength(c, True)
        temp = 600.00
        # 周长面积筛选
        if M['m00'] > temp and perimeter < 8000:
            # 周长和面积比值筛选
            if (M['m00'] / perimeter) > 3.5:
                # 建立掩膜，
                mask = np.zeros(canny.shape, np.uint8)
                cv2.drawContours(mask, [c], 0, 255, -1)
                # cv2.imshow('mask', mask)
                # cv2.waitKey()
                # 查找掩膜区域内所有的点集
                pixelpoints = np.transpose(np.nonzero(mask))
                # 将符合条件的掩膜区域进行统计像素点判断颜色
                color = judge_color(img, pixelpoints)
                if color == True:
                    return True
    return False


def judge_open_close(img):
    # 建议hsv取值范围
    red1_lower = np.array([150, 43, 46])  # 红色下限
    red1_upper = np.array([180, 255, 255])  # 红色上限
    red2_lower = np.array([0, 43, 46])  # 红色下限
    red2_upper = np.array([10, 255, 255])  # 红色上限
    green_lower = np.array([50, 20, 10])
    green_upper = np.array([100, 255, 255])
    lower = [red1_lower, red2_lower, green_lower]

    upper = [red1_upper, red2_upper, green_upper]

    img = cv2.resize(img, (250, 450), )  # 统一大小
    blur = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波

    # 根据阈值范围提取颜色判断结果
    img_result = color_trace(lower, upper, blur)
    if img_result == True:
        print('method1')
        return True
    else:
        print('method2')
        # 提取轮廓，判断轮廓内颜色判断结果
        judge_lk = judge_lunkuo(img)
        if judge_lk == True:
            return True
        else:
            return False

def respirator_color(img):
    img = cv2.resize(img, (250, 450), )  # 统一大小
    num=0
    #num1 = 0
    red1_lower = np.array([150, 43, 46])  # 红色下限
    red1_upper = np.array([180, 255, 255])  # 红色上限
    # red2_lower = np.array([0, 43, 46])  # 红色下限
    # red2_upper = np.array([10, 255, 255])  # 红色上限

    hsvs=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    height = img.shape[0]
    weight = img.shape[1]

    for row in range(height):  # 遍历高
        for col in range(weight):
            #num1+=1
            hsv=hsvs[row,col]
            if red1_lower[0]<=hsv[0]<=red1_upper[0] : #or  red2_lower[0]<=hsv[0]<=red2_upper[0] :
                if red1_lower[1]<=hsv[1]<=red1_upper[1] : #or  red2_lower[1]<=hsv[1]<=red2_upper[1] :
                    if red1_lower[2]<=hsv[2]<=red1_upper[2] : #or  red2_lower[2]<=hsv[2]<=red2_upper[2] :
                        num+=1
    #print(num,num1)
    if num > 1000:
        #print(num)
        return True
    else:
        return False

if __name__ == "__main__":
    '''True:green,False:red'''
    # file = '/home/linxu/Documents/Work/sources/状态指示器图片/16.png'
    # file = '/home/linxu/Desktop/1_0_0_1020_1_0/73849_steerPoint61_preset819_20210512162504_v.jpeg'
    # file = '/home/linxu/Desktop/1_0_0_1020_1_0/66241_steerPoint49_preset803_20210510180651_v.jpeg'
    # file = '/home/linxu/Desktop/1_0_0_1020_1_0/62806_steerPoint53_preset806_20210604160322_v.jpeg'

    # file = '/home/linxu/Desktop/1_0_0_1020_1_0/26273_steerPoint20_preset1128_20210604170647_v.jpeg'

    # file = '/home/linxu/Desktop/1_0_0_1020_1_0/106168_steerPoint15_preset737_20210510153521_v.jpeg'

    # file = '/home/linxu/Desktop/1_0_0_1020_1_0/109832_steerPoint53_preset806_20210509090051_v.jpeg'
    file = '/home/linxu/Documents/Work/testpic/状态指示器/Green/9.jpeg'

    img = cv2.imread(file)
    cv2.imshow('img', img)
    cv2.waitKey()
    is_true = judge_open_close(img)
    print(is_true)
    # 17
    # import os
    # path = r"C:\temptest\work\wgs\1_0_0_10\white\90.jpg"
    # img = cv2.imread(os.path.join(path))
    # img_result = respirator_color(img)
    # # ture:红，false：白
    # if img_result == True:
    #     print("红")
    # # 若没有被定义为绿色，且没有符合条件的轮廓，则进行轮廓法判断
    # else:
    #     print("白")