# -*- coding: utf-8 -*-
#
# 文字角度相关函数
# Author: alex
# Created Time: 2020年01月03日 星期五 18时36分08秒
import cv2
import numpy as np
from scipy.ndimage import filters, interpolation
from image_utils.utils import conc_map


def estimate_skew_angle(gray, fine_tune_num=4, step_start=0.75,
                        max_workers=None, scale=600., max_scale=900.):
    """
    估计图像文字角度
    :param gray 待纠正的灰度图像
    :param fine_tune_num 微调的次数, 界定了微调的精度
        当该值为n时，表示微调角度精确到step_start乘以10的-(n-1)次方
    :param step_start 步长的初始值
        当该值为a时，其纠正的角度范围是[-10*a, 10*a]。该值不应该大于4.5
    :param max_workers int|None 并发的进程数量限制
    :param scale, max_scale float 计算时缩放的最小最大宽高
    :return angle 需要纠正的角度
    """
    def resize_im(im, scale, max_scale):
        f = scale / min(im.shape[:2])
        max_rate = max_scale / max(im.shape[:2])
        f = min(f, max_rate)
        return cv2.resize(im, (0, 0), fx=f, fy=f)

    gray = resize_im(gray, scale, max_scale)
    g_min, g_max = np.amin(gray), np.amax(gray)
    if g_max - g_min < 30:
        return 0.
    # 归一化
    image = (gray-g_min) / (g_max-g_min)
    m = interpolation.zoom(image, 0.5)
    m = filters.percentile_filter(m, 80, size=(20, 2))
    m = filters.percentile_filter(m, 80, size=(2, 20))
    m = interpolation.zoom(m, 1.0/0.5)

    w, h = min(image.shape[1], m.shape[1]), min(image.shape[0], m.shape[0])
    flat = np.clip(image[:h, :w]-m[:h, :w]+1, 0, 1)
    d0, d1 = flat.shape
    o0, o1 = int(0.1*d0), int(0.1*d1)
    flat = np.amax(flat)-flat
    flat -= np.amin(flat)
    est = flat[o0:d0-o0, o1:d1-o1]

    angle, step = 0, step_start   # 纠正角度的初始值和步长
    for _ in range(fine_tune_num):
        angle = fine_tune_angle(est, step, start=angle,
                                max_workers=max_workers)
        step /= 10

    return angle


def fine_tune_angle(image, step, start=0, max_workers=None):
    """微调纠正
    在某个角度start的周围进行微调
    """
    def var(i):
        # 从-10到10
        angle = start + (i-5)*step
        roest = interpolation.rotate(image, angle, order=0, mode='constant')
        v = np.mean(roest, axis=1)
        v = np.var(v)
        return (v, angle)

    estimates = conc_map(var, range(11), max_workers=max_workers)
    _, angle = max(estimates)
    return angle


if __name__ == '__main__':
    import sys
    from convert import rotate
    img = cv2.imread(sys.argv[1], cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(img)
    print(angle)
    new_img = rotate(img, angle)
    cv2.imwrite(sys.argv[2], new_img)
