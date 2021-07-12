'''
@author: linxu
@contact: 17746071609@163.com
@time: 2020-05-24 上午10:24
@desc: 视觉图像增强技术
'''

import cv2
import numpy as np

def subtractGauss(src, sigmaX=2.0, sigmaXX=2.2):
    '''
    高斯阈值滤波
    :param src:
    :param sigmaX:
    :param sigmaXX:
    :return:
    '''
    image_gray = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2GRAY)
    gauss20 = cv2.GaussianBlur(src=image_gray, ksize=(0, 0), sigmaX=sigmaX)
    gauss22 = cv2.GaussianBlur(src=image_gray, ksize=(0, 0), sigmaX=sigmaXX)
    subtract = cv2.subtract(src1=gauss22, src2=gauss20, dtype=cv2.CV_32F)
    subtract = np.uint8(subtract)
    return subtract



def blackoutlineenhancement(img, low=120, upper=255):
    '''
    强化图中黑线条
    :param img:
    :param low:
    :param upper:
    :return:
    '''
    size = (img.shape[0], img.shape[1])
    _, threshold_img = cv2.threshold(
        cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), low, upper,
        cv2.THRESH_BINARY)
    return cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2BGR)


def his_equl_color(img):
    '''
    直方图均衡
    :param img:
    :return:
    '''
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])  # equalizeHist(in,out)
    cv2.merge(channels, ycrcb)
    img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq

# 视网膜-大脑皮层(Retinex)增强算法
def single_scale_retinex(img, sigma):
    temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(temp == 0, 0.01, temp)
    retinex = np.log10(img + 0.01) - np.log10(gaussian)
    return retinex

def multi_scale_retinex(img, sigma_list):
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        retinex = single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration_temp = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration_temp

def multi_scale_retinex_with_color_restoration(img, sigma_list, g, b, alpha, beta):
    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_msrcr = g * (img_retinex * img_color + b)
    return img_msrcr

def touint8(img):
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / \
                       (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255
    img = np.uint8(np.minimum(np.maximum(img, 0), 255))
    return img

def simplest_color_balance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]

    low_val = 0.0
    high_val = 0.0

    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0

        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img

'''
Multi Scale Retinex With Color Restoration 1
多尺度视网膜与颜色恢复
'''
def m_s_r_c_r(img, sigma_list, g=5, b=25, alpha=125, beta=46, low_clip=0.01, high_clip=0.99):
    if sigma_list is None:
        sigma_list = [15, 80, 250]
    msrcr = multi_scale_retinex_with_color_restoration(img, sigma_list, g, b, alpha, beta)
    msrcr = touint8(msrcr)
    msrcr = simplest_color_balance(msrcr, low_clip, high_clip)
    return msrcr

if __name__ == '__main__':
    img_path = '../../../images/表计/test3.png'
    src = cv2.imread(img_path)
    cv2.imshow('src', src)
    cv2.waitKey()
    # 算法1:高斯阈值滤波
    dst1 = subtractGauss(src, 2.0, 2.2)
    cv2.imshow('dst1', dst1)
    cv2.waitKey()
    # 算法2:二值化增强黑色轮廓
    dst2 = blackoutlineenhancement(src, 100,255)
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    # 算法3:直方图均衡化
    dst3 = his_equl_color(src)
    cv2.imshow('dst3', dst3)
    cv2.waitKey()
    # 算法4:多尺度视网膜与颜色恢复
    dst4 = m_s_r_c_r(src, sigma_list=[15, 80, 250])
    cv2.imshow('dst4', dst4)
    cv2.waitKey()

