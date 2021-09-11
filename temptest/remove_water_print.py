# 方式二
import cv2
import numpy


def get_water2():
    src = cv2.imread('yuan.png')
    mask = cv2.imread('di2.png')
    save = numpy.zeros(src.shape, numpy.uint8)  # 创建一张空图像用于保存

    for row in range(src.shape[0]):
        for col in range(src.shape[1]):
            for channel in range(src.shape[2]):
                if mask[row, col, channel] == 0:
                    val = 0
                else:
                    reverse_val = 255 - src[row, col, channel]
                    val = 255 - reverse_val * 256 / mask[row, col, channel]
                    if val < 0: val = 0

                save[row, col, channel] = val

    cv2.imwrite('result2.jpg', save)


get_water2()
