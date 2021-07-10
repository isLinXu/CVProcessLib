import cv2
import matplotlib.pyplot as plt

imagepath = r'/home/linxu/Desktop/武高所图片测试/状态指示器/5.png'
image = cv2.imread(imagepath)
height, width, channel = image.shape
for i in range(height):
    for j in range(width):
        b, g, r = image[i, j]
        if (r > (30 +b)  and r > (30+ g)):  # 对蓝色进行判断，30不错
            b = 0
            g = 0
            r = 0
        else:
            b = 255
            g = 255
            r = 255

        image[i, j] = [r, g, b]
plt.figure(figsize=(20, 10))
plt.imshow(image)
plt.show()