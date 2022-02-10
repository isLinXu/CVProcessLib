
# 样例：语义分割数据集抽样可视化
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

image_path_list = ['samples/images/H0002.jpg', 'samples/images/H0005.jpg']
label_path_list = [path.replace('images', 'labels').replace('jpg', 'png')
                   for path in image_path_list]

plt.figure(figsize=(8, 8))
for i in range(len(image_path_list)):
    plt.subplot(len(image_path_list), 2, i*2+1)
    plt.title(image_path_list[i])
    plt.imshow(cv2.imread(image_path_list[i])[:, :, ::-1])

    plt.subplot(len(image_path_list), 2, i*2+2)
    plt.title(label_path_list[i])
    plt.imshow(cv2.imread(label_path_list[i], cv2.IMREAD_GRAYSCALE))
plt.tight_layout()
plt.show()