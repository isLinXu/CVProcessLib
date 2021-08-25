# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_gray_kmean(img):
    # 获取图像高度、宽度
    rows, cols = img.shape[:]

    # 图像二维像素转换为一维
    data = img.reshape((rows * cols))
    data = np.float32(data)

    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS

    # K-Means聚类 聚集成4类
    compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

    # 生成最终图像
    res = centers[labels.flatten()]
    dst = res.reshape((img.shape[0], img.shape[1]))
    print(dst)
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    # 用来正常显示中文标签img
    # plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图像
    titles = [u'原始图像', u'聚类图像']
    images = [img, dst]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()





if __name__ == '__main__':
    file_path = '/home/hxzh02/文档/defectDetect/金属锈蚀(复件)/src/32.jpg'
    # 读取原始图像灰度颜色
    img = cv2.imread(file_path, 0)
    cv2.imshow('img',img)
    cv2.waitKey()
    # print(img.shape)
    color_gray_kmean(img)
