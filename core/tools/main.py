import cv2


from core.tools.image_filtering import show_filtering
from core.tools.image_outline import show_outline
from core.tools.image_transformation import show_transformation
from core.tools.utils import plt_save
from core.tools.image_color import show_hsv
from core.tools.image_enhancement import show_enhancement

import os, sys
# 基本路径
RUN_PATH = os.path.dirname(__file__)

if __name__ == '__main__':
    print(RUN_PATH)
    file_path = RUN_PATH+'/image/test.png'
    # file_path = '/home/linxu/opencv_tools-main/images/test1.png'

    origin = cv2.imread(file_path)
    origin = origin[:, :, [2, 1, 0]]

    # x, y = origin.shape[0:2]
    # origin = cv2.resize(origin, (int(y / 3), int(x / 3)))
    plt_save(image=origin, title='Origin')

    # --------------------图像色彩--------------------
    # 转换成HSV色彩空间
    show_hsv(origin)

    # --------------------图像变换--------------------
    show_transformation(origin)

    # --------------------图像过滤--------------------
    show_filtering(origin)

    # --------------------提取直线、轮廓、区域--------------------
    show_outline(origin)

    # -------------------- 图像增强-白平衡等--------------------
    show_enhancement(origin)

    print('done')
