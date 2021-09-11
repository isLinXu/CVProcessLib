'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-05-26 上午11:54
@desc: 颜色分割
'''

import cv2
import numpy as np
import os

def color_cut(img):
    # Step1. 转换为HSV
    hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 用颜色分割图像
    low_range = np.array([20, 90, 27])
    high_range = np.array([134, 255, 255])
    th = cv2.inRange(hue_image, low_range, high_range)
    return th


if __name__ == '__main__':
    # file_path = '/home/hxzh02/文档/defectDetect/金属锈蚀(复件)/src/demo/7.jpg'
    file_path = '/home/hxzh/Wei_Work/Project/Chopper/yolov5_Dlq/crop/001.jpg'

    # file_path = '/home/hxzh02/文档/defectDetect/金属锈蚀(复件)/src/124.jpg'
    img = cv2.imread(file_path)
    cv2.imshow('img',img)
    print(img.shape)
    # 颜色分割
    th = color_cut(img)
    cv2.imshow('th', th)
    # cv2.imwrite('/home/hxzh02/文档/defectDetect/金属锈蚀(复件)/src/demo/' + '124dst.png', th)
    cv2.waitKey()

    # path = '/home/hxzh02/文档/defectDetect/金属锈蚀(复件)/src/'
    # for root, dirs, files in os.walk(path):
    #     for name in files:
    #         if len(dirs) != 0:
    #             fname = os.path.join(root, name)
    #             print('fname', fname)
    #             print('root', root)
    #             print('name', name)
    #             img = cv2.imread(fname)
    #             cv2.imshow('img',img)
    #             print(img.shape)
    #             # 颜色分割
    #             th = color_cut(img)
    #             cv2.imshow('th', th)
    #             # cv2.imwrite(root + 'demo/dst/' + 'dst_1' + name, th)
    #             # cv2.waitKey()





