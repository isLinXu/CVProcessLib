#!/usr/bin/env python
# coding: utf-8

# Import necesssary libraries to extract the temperature per pixel
import flirimageextractor
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2


def calc_Temp(file_path):
    # Read the image using OpenCV library
    image = cv2.imread(file_path)
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.show()

    # Find the shape of the image
    rows, cols, color = image.shape
    print(image.shape)

    # Process the image using FlirImageExtractor library
    flir = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
    flir.process_image(file_path)
    flir.save_images()
    flir.plot()
    thermal = flir.get_thermal_np()
    flir.check_for_thermal_image(file_path)

    # Extract the temperature per pixel of the image Transformer.jpg
    for i in range(image.shape[0]):
        for j in range(0, image.shape[1]):
            pixel = image[i, j]
            temperature = thermal[i, j]
            print(pixel, temperature)


if __name__ == '__main__':
    # file_path = '/home/hxzh02/文档/Transformer_thermal_Detection/Transformer.jpg'
    file_path = '/media/hxzh02/7A50-5158/dji_thermal_sdk_v1.0_20201110/dataset/XTS/DJI_0002_R.jpg'
    # file_path = '/media/hxzh02/7A50-5158/红外图像检测网盘资料/红外图片素材/红外测温图像素材1/FLIR4487.jpg'
    calc_Temp(file_path)





