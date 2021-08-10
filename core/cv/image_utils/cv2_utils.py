'''
cv2相关函数

Author: alex
Created Time: 2020年11月03日 星期二 09时44分42秒
'''
import cv2


def lapulase(gray):
    """计算拉普拉斯算子：图像模糊度
    注意：
    1. 在比较模糊度的时候，图像应该resize到相同的大小
    2. 返回的得分阈值很重要
    @param gray cv2灰度图像
    @return 清晰度得分 该值越大通常越清晰
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()
