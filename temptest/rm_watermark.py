'''


Author: alex
Created Time: 2020年08月20日 星期四 16时09分37秒
'''
import cv2
import numpy as np


def remove_watermark(image, thr=200, convol=3):
    """
    简单粗暴去水印，可将将pdf或者扫描件中水印去除
    使用卷积来优化计算
    :param image: 输入图片，cv格式灰度图像
    :param thr:   去除图片中像素阈值
    :param convol: 卷积窗口的大小
    :return: 返回np.array格式图片
    """
    distance = int((convol - 1) / 2)     # 为了执行卷积，对图像连缘进行像素扩充
    # 使用白色来进行边缘像素扩充
    image = cv2.copyMakeBorder(image, distance, distance, distance, distance,
                               cv2.BORDER_CONSTANT, value=255)
    mask = (image < 200).astype(int)
    # 单位矩阵卷积操作
    mask = cv2.boxFilter(mask, -1, (convol, convol), normalize=False)
    mask = (mask >= 1).astype(int)     # 掩膜构建完成，>=1表示窗口内有黑点
    image[np.where(mask == 0)] = 255   # 掩膜中为0的位置赋值为255，白色,达到去水印效果
    h, w = image.shape[:2]
    image = image[distance:h - distance, distance:w - distance]
    return image


def bak_remove_watermark(image, thr=200, distance=1):
    """
    简单粗暴去水印，可将将pdf或者扫描件中水印去除
    :param image: 输入图片，Image格式
    :param thr: 去除图片中像素阈值
    :param distance: 去除图片中像素距离
    :return: 返回np.arrayg格式图片
    """
    w, h = image.size
    rgb_im = image.convert('RGB')
    for x in range(0, w - 1):
        for y in range(0, h - 1):
            if not hasBlackAround(x, y, distance, rgb_im, thr=thr):
                rgb_im.putpixel((x, y), (255, 255, 255))

    return rgb_im


def hasBlackAround(x, y, distance, img, thr=200):
    w, h = img.size
    startX = max(0, x-distance)
    startY = max(0, y-distance)
    endX = min(w-1, x+distance)
    endY = min(h-1, y+distance)
    for j in range(startX, endX):
        for k in range(startY, endY):
            r, g, b = img.getpixel((j, k))
            if r < thr and g < thr and b < thr:
                # 满足条件的点黑点
                return True

    return False


if __name__ == '__main__':
    from PIL import Image
    debug = False
    image_path = "gf-png/gf1.png"
    img = Image.open(image_path)
    res_img = remove_watermark(img, thr=100, distance=1)
