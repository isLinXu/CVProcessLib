#!/usr/bin/python3
import numpy as np
import cv2

DEFAULT_CARTOONIFY = {
    "bilateral_stages": 7,
    "bilateral_diameter": 9,
    "bilateral_sigma_color": 9,
    "bilateral_sigma_space": 7,
    "blur_kernal_size":9,
    "adaptive_thresh_block": 9,
    "adaptive_thresh_const": 2,
}

def cartoonify(filename, out_size, out_file="/tmp/cartoon.png", **kwargs):
    """将输入信号卡通化
        减少颜色级别的数量，去除细节和高光某些边缘。
        Note
        -----
        根据参考资料，使用双边滤波器来平滑图像并减少色阶。
        然后应用边缘检测应用前的图像灰度和自适应检测模糊图像上的增强边缘。
        参数
        ----------
        文件名：str
            输入图像的路径
        out_size : 元组
            输出图像大小
        输出文件：str
            输出文件名
        """

    # -------------  load from keyword arguments or use default  ------------
    n_bilat = kwargs.get("bilateral_stages", 7)
    bilat_diameter = kwargs.get("bilateral_diameter", 9)
    bilat_sigma_color = kwargs.get("bilateral_sigma_color", 9)
    bilat_sigma_space = kwargs.get("bilateral_sigma_space", 7)
    blur_kernal_size = kwargs.get("blur_kernal_size", 9)
    adapt_threshold_block = kwargs.get("adaptive_thresh_block", 9)
    adapt_threshold_const = kwargs.get("adaptive_thresh_const", 2)

    # ------------------  load image and downsample  ------------------------
    image_rgb = cv2.imread(filename)

    # identify the downsample
    down = np.max((int(image_rgb.shape[0] / out_size[0]),
        int(image_rgb.shape[1] / out_size[1])))

    # downsample by Gaussian pyramid
    for _ in range(down):
        image_rgb = cv2.pyrDown(image_rgb)

    # ------------------------  bilateral filter  ---------------------------
    for _ in range(n_bilat):
        image_rgb = cv2.bilateralFilter(image_rgb, d=bilat_diameter,
            sigmaColor=bilat_sigma_color, sigmaSpace=bilat_sigma_space)

    # -------------------------  median filter  -----------------------------
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, blur_kernal_size)

    # ------------------------  enhance edges  ------------------------------
    edges = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blockSize=adapt_threshold_block, C=adapt_threshold_const)

    # convert back to color
    edges  = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # enchance edges
    cartoon_image = cv2.bitwise_and(image_rgb, edges)

    # -------------------------  save image  --------------------------------
    cv2.imwrite(out_file, cartoon_image)
    cv2.imshow('src',image_rgb)
    cv2.imshow('dst', cartoon_image)
    cv2.waitKey()


if __name__ == "__main__":
    file_name = '/home/hxzh02/PycharmProjects/cvprocess-lib/images/lena.png'
    cartoonify(file_name,(800, 600))
