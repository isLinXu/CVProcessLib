'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-08-20 上午11:54
@desc: aruco_marker生成器
'''

import cv2
import numpy as np

def aruco_maker():
    """
    生成aruco-marker标记
    :return:
    """
    # 加载预定义的字典
    '''
    DICT_4X4_50,100,250,1000
    DICT_5X5_50,100,250,1000
    DICT_6X6_50,100,250,1000
    DICT_7X7_50,100,250,1000
    '''
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    # 配置相关信息
    # size大小
    size = (200, 200)
    # 建立模版
    markerImage = np.zeros(size, dtype=np.uint8)
    # 设置ID
    id = 0
    # 设置像素大小
    sidePixels = 500
    # 设置mm单位(转换关系:1像素约等于3分之一毫米)
    mmPixels = sidePixels * 0.33
    # 绘制图形生成标记
    markerImage = cv2.aruco.drawMarker(dictionary, id, sidePixels, markerImage, 1)
    # # 存为文件
    # cv2.imshow('markerImage',markerImage)
    # cv2.waitKey()
    # cv2.imwrite("marker22.png", markerImage)
    return markerImage

def aruco_automaker(dictionary,size = (200, 200),id = 0,sidePixels = 500):
    """
    配置生成aruco-marker标记
    :return:
    """
    # 加载预定义的字典
    '''
    DICT_4X4_50,100,250,1000
    DICT_5X5_50,100,250,1000
    DICT_6X6_50,100,250,1000
    DICT_7X7_50,100,250,1000
    '''
    # 建立模版
    markerImage = np.zeros(size, dtype=np.uint8)
    # 设置mm单位(转换关系:1像素约等于3分之一毫米)
    mmPixels = sidePixels * 0.33
    # 绘制图形生成标记
    markerImage = cv2.aruco.drawMarker(dictionary, id, sidePixels, markerImage, 1)
    # # 存为文件
    # cv2.imshow('markerImage',markerImage)
    # cv2.waitKey()
    # cv2.imwrite("marker22.png", markerImage)
    return markerImage



if __name__ == '__main__':

    # # 生成aruco-marker标记
    # markerImage = aruco_maker()

    # 加载预定义的字典
    '''
    DICT_4X4_50,100,250,1000
    DICT_5X5_50,100,250,1000
    DICT_6X6_50,100,250,1000
    DICT_7X7_50,100,250,1000
    '''
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    # 配置相关信息
    # size大小
    size = (200, 200)
    # 设置ID
    id = 0
    id_max = 14
    idx = 1
    # 设置像素大小
    sidePixels = 500
    tag = True

    if (tag):
        # 遍历批量生成
        for idx in range(id,id_max):
            markerImage = aruco_automaker(dictionary,size,idx,sidePixels)
            cv2.imshow('markerImage', markerImage)
            cv2.imwrite("marker" + str(idx) + '.png', markerImage)
            cv2.waitKey()
    else:
        # 单张生成
        markerImage = aruco_automaker(dictionary, size, idx, sidePixels)
        cv2.imshow('markerImage', markerImage)
        cv2.imwrite("marker" + str(idx) + '.png', markerImage)
        cv2.waitKey()