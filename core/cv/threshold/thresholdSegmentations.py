'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-06-29 下午19:07
@desc: 针对五种不同转换方式，采用滑动条实现动态全局阈值分割

cv2.threshold(src, thresh, maxval, type)
参数：
    src:输入的图像
    thresh:图像分割所用的阈值（threshold value）
    maxval:当阈值类型（thresholding type）采用cv2.THRESH_BINARY和cv2.THRESH_BINARY_INV时像素点被赋予的新值
    type:介绍5种转换方式：
        cv2.THRESH_BINARY（当图像某点像素值大于thresh(阈值)时赋予maxval，反之为0。注：最常用且为默认值）
        cv2.THRESH_BINARY_INV（当图像某点像素值小于thresh时赋予maxval，反之为0）
        cv2.THRESH_TRUNC（当图像某点像素值大于thresh时赋予thresh,反之不变。注：虽然maxval没用了，但是调用函数不能省略）
        cv2.THRESH_TOZERO（当图像某点像素值小于thresh时赋予0，反之不变。注：同上）
        cv2.THRESH_TOZERO_INV（当图像某点像素值大于thresh时赋予0，反之不变。注：同上）
   补：cv2.THRESH_OTSU（该方法自动寻找最优阈值，并返回给retval，见下文。实际使用时配合上述5种转变方式之一，例如：cv2.THRESH_OTSU+cv2.THRESH_BINARY）
'''

import cv2
import numpy as np

def threshold_segmentation(thresh):
    '''
    创建滑动条回调函数，参数thresh为滑动条对应位置的数值
    :param thresh:
    :return:
    '''
    #采用5种阈值类型（thresholding type）分割图像
    retval1,img_binary=cv2.threshold(img_original,thresh,255,cv2.THRESH_BINARY)
    retval2,img_binary_invertion=cv2.threshold(img_original,thresh,255,cv2.THRESH_BINARY_INV)
    retval3,img_trunc=cv2.threshold(img_original,thresh,255,cv2.THRESH_TRUNC)
    retval4,img_tozero=cv2.threshold(img_original,thresh,255,cv2.THRESH_TOZERO)
    retval5,img_tozero_inversion=cv2.threshold(img_original,thresh,255,cv2.THRESH_TOZERO_INV)
    # 由于cv2.imshow()显示的是多维数组（ndarray）,因此我们通过np.hstack(数组水平拼接)
    # 和np.vstack(竖直拼接)拼接数组，达到同时显示多幅图的目的
    img1 = np.hstack([img_original,img_binary,img_binary_invertion])
    img2 = np.hstack([img_trunc,img_tozero,img_tozero_inversion])
    global imgs
    imgs=np.vstack([img1,img2])

if __name__ == '__main__':
    # 载入原图，转化为灰度图像,并通过cv2.resize()等比调整图像大小
    img_original=cv2.imread('/home/linxu/PycharmProjects/CVProcessLib/images/数显表/1.jpeg')
    img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    # img_original=cv2.resize(gray,(0,0),fx=0.3,fy=0.3)
    # 初始化阈值，定义全局变量imgs
    thresh=130
    imgs=0

    # 新建窗口
    cv2.namedWindow('Images')
    # 新建滑动条，初始位置为130
    cv2.createTrackbar('threshold value','Images',130,255,threshold_segmentation)
    # 第一次调用函数
    threshold_segmentation(thresh)
    # 显示图像
    while(1):
        cv2.imshow('Images',imgs)
        # 按下q键退出
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows()