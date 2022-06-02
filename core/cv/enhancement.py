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


if __name__ == '__main__':
    img_path = 'test1.png'
    src = cv2.imread(img_path)
    cv2.imshow('src', src)
    cv2.waitKey()
    # 算法1:高斯阈值滤波
    dst1 = subtractGauss(src, 2.0, 2.2)
    cv2.imshow('dst1', dst1)
    cv2.waitKey()
    # 算法2:二值化增强黑色轮廓
    dst2 = blackoutlineenhancement(src, 120,255)
    cv2.imshow('dst2', dst2)
    cv2.waitKey()
    # 算法3:直方图均衡化
    dst3 = his_equl_color(src)
    cv2.imshow('dst3', dst3)
    cv2.waitKey()
