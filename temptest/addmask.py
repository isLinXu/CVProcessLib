import cv2
import numpy as np

# src = '/home/hxzh02/文档/无人机数据集整理/杆塔图片汇总/src/13.jpg'
# mask = '/home/hxzh02/PytorchLab/unet-flask-line/output_13.png'
# src = '/home/hxzh02/下载/PPLTA航拍输电线路数据集/data_original_size/imgs/74_00765.png'
# mask = '/home/hxzh02/PytorchLab/unet-flask-line/output_74_00765.jpg'
src = '/home/hxzh02/PytorchLab/unet-flask-line/data/imgs/63_01901.png'
mask = '/home/hxzh02/PytorchLab/unet-flask-line/output_63_01901.jpg'
# src = "/home/hxzh02/PytorchLab/unet-flask-line/data/imgs/66_00136.png"
# mask = "/home/hxzh02/PytorchLab/unet-flask-line/output_66_00136.jpg"

# 使用opencv叠加图片
img1 = cv2.imread(src)
img2 = cv2.imread(mask)

alpha = 1
meta = 0.4
gamma = 0
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
image = cv2.addWeighted(img1, alpha, img2, meta, gamma)

cv2.imshow('image', image)
cv2.waitKey(0)