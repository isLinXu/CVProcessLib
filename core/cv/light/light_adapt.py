import numpy as np
import cv2
import math
def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel
def aug(src):
    """图像亮度增强"""
    if get_lightness(src) > 128:
        print("图片亮度足够，不做增强")
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)
    return out

def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    aaa=hsv_image[:, :, 2]
    lightness = hsv_image[:, :, 2].mean()
    return lightness


def gamma_trans(img):  # gamma函数处理
    img_gray=cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
    gamma_table = [np.power(x / 255.0, gamma_val) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

if __name__ == "__main__":
    img = cv2.imread(r"/home/hxzh/下载/20190315110557148.png")
    img = aug(img)
    image_gamma_correct = gamma_trans(img)  # gamma变换
    # cv2.imwrite('out.jpg', img)
    cv2.imshow("out",img)
    cv2.imshow("image_gamma_correct", image_gamma_correct)
    cv2.waitKey(0)
