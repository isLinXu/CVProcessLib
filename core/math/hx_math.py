import numpy as np
from scipy.optimize import leastsq


######最小二乘法#####
# 拟合函数
def func(a, x):
    k, b = a
    return k * x + b


# 残差
def dist(a, x, y):
    return func(a, x) - y
    #  求seg_img图中的直线与垂直方向的夹角


def fitting(x_list, y_list):
    '''最小二乘法拟合直线
    x_list:所有x坐标点
    y_list:所有y坐标点
    len(x_list)==len(y_list)'''
    param = [0, 0]
    var = leastsq(dist, param, args=(np.array(x_list), np.array(y_list)))
    k, b = var[0]
    return k, b


######已知坐标两点求K,B直线方程#####
def line_kb(max_x, max_y, min_x, min_y):
    if (max_x - min_x)==0 or ((max_x - min_x)<0.1 and (max_x - min_x)>-0.1 ):
        return 0,0
    k = (max_y - min_y) / (max_x - min_x)
    b = min_y - k * min_x
    return k, b



######已知一组坐标点及y最大值求当y最大时x最大或最小值#####
def find_minx(y_list, y_value, x_list):
    finally_pos = 0
    min_x = 10000
    for i in range(y_list.count(y_value)):
        next_pos = y_list.index(y_value)
        x = x_list[finally_pos + next_pos]
        if min_x > x:
            min_x = x
        first_pos = next_pos + 1
        finally_pos += first_pos
        y_list = y_list[first_pos:]
    return min_x


def find_maxy(y_list, y_value, x_list):
    finally_pos = 0
    min_x = 0
    for i in range(y_list.count(y_value)):
        next_pos = y_list.index(y_value)
        x = x_list[finally_pos + next_pos]
        if min_x < x:
            min_x = x
        first_pos = next_pos + 1
        finally_pos += first_pos
        y_list = y_list[first_pos:]
    return min_x
def hx_circle(x1, y1, x2, y2, x3, y3):
    '''已知圆上三点坐标,得出圆心和半径'''
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    a1 = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0
    a2 = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0
    theta = b * c - a * d;
    if abs(theta) < 1e-7:
        return -1
    x0 = (b * a2 - d * a1) / theta;
    y0 = (c * a1 - a * a2) / theta;
    r = np.sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2))
    return x0,y0,r


if __name__ == "__main__":
    ######最小二乘法#####
    x_list = [0, 1, 2, 3, 1.2, 2.2, 3.2]
    y_list = [0, 1, 2, 3, 1, 2, 3]
    k, b = fitting(x_list, y_list)
    print("最小二乘法:", k, b)
    ######已知坐标两点求K,B直线方程#####
    k, b = line_kb(2, 2, 1, 1.2)
    print("直线方程:", k, b)
