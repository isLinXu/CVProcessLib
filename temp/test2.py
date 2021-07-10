# 基于聚类K-Means方法实现图像分割
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans


# Define loadDato to solve my image
def loadData(filePath):
    f = open(filePath, 'rb')  # deal with binary
    data = []
    img = image.open(f)  # return to pixel(像素值)
    m, n = img.size  # the size of image
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            # deal with pixel to the range 0-1 and save to data
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return np.mat(data), m, n

# /home/linxu/Desktop/武高所图片测试/状态指示器/5.png
imgData, row, col = loadData("/home/linxu/Desktop/武高所图片测试/状态指示器/5.png")
# setting clusers(聚类中心) is 3
label = KMeans(n_clusters=3).fit_predict(imgData)
# get the label of each pixel
label = label.reshape([row, col])
# create a new image to save the result of K-Means
pic_new = image.new("L", (row, col))
# according to the label to add the pixel
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
pic_new.save("/home/linxu/Desktop/武高所图片测试/状态指示器/5km.png", "JPEG")