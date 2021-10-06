
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def ImgProcessionShape(img):
    """图像规整"""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] > 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def ImgConvertL(img):
    """自己写的灰度化图像"""
    # 模拟pillow的L模式，可用numpy改进速度，先保持跟C++代码一致
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_gray = np.array((0.299 * r + 0.587 * g + 0.114 * b), dtype=np.uint8)
    new_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    return new_img

def ColorEnhanceC(img, color=1.5):
    """色度增强"""
    img = ImgProcessionShape(img)
    new_img = img.copy()
    img_R = img.copy()
    # 模拟pillow的色度增强，可用numpy改进速度，先保持跟C++代码一致
    img_L = ImgConvertL(new_img)
    # 融合图像
    new_img = cv2.addWeighted(img_L, 1.0 - color, img_R, color, 0)
    return new_img

def ContrastEnhance(img, contrast=1.5):
    """对比度增强"""
    # PIL的对比度增强 因为无法跟C++兼容 所以使用自己写的算法
    img = ImgProcessionShape(img)
    image_temp = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enh_con = ImageEnhance.Contrast(image_temp)
    image_contrasted = enh_con.enhance(contrast)
    new_img = cv2.cvtColor(np.asarray(image_contrasted), cv2.COLOR_RGB2BGR)
    return new_img

