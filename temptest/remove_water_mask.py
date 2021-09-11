import os
from PIL import Image
import numpy as np
import imghdr
import cv2

# 创建文件夹
def mkdir(path):
    path = path.strip().rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

# 递归访问文件/文件夹
def visit_dir_files(org_input_dir, org_output_dir, recursion_dir, img2):
    single_file = False
    if os.path.isdir(recursion_dir):
        dir_list = os.listdir(recursion_dir)
    else:
        dir_list = [recursion_dir]
        single_file = True
    for i in range(0, len(dir_list)):
        path = os.path.join(recursion_dir, dir_list[i])
        if os.path.isdir(path):
            visit_dir_files(org_input_dir, org_output_dir, path)
        else:
            if imghdr.what(path):
                abs_output_dir = org_output_dir + recursion_dir[len(org_input_dir):]
                target_path = os.path.join(abs_output_dir, dir_list[i])
                if single_file:
                    target_path = os.path.join(org_output_dir, os.path.basename(dir_list[i]))
                target_dirname = os.path.dirname(target_path)
                if not os.path.exists(target_dirname):
                    mkdir(target_dirname)
                img_deal(path, target_path, img2)

# 图片处理``
def img_deal(img_path, save_path, img2):
    img = Image.open(img_path)
    img = levelsDeal(img, img2)
    img_res = Image.fromarray(img.astype('uint8'))
    print(u'图片[' + img_path + u']处理完毕')
    img_res.save(save_path)


# 图像矩阵处理
def levelsDeal(img, img2):
    thresh = cv2.inRange(img2, np.array([40]), np.array([200]))
    scan = np.ones((5, 5), np.uint8)
    cor = cv2.dilate(thresh, scan, iterations=1)
    img_array = np.array(img, dtype=int)
    h1, w1, _ = img_array.shape
    h2, w2 = cor.shape
    h = min(h1,h2)
    w = min(w1, w2)
    print(cor.shape)
    img_array = img_array[:h,:w,:1].reshape(h, w)
    cor = cor[:h,:w]
    # 找出有水印的地方，有水印的为1
    remove_watermark = cor/255*img_array
    # 利用色差去除水印，根据实际情况自己修改
    #img_array = np.minimum(img_array, 100).astype(np.uint8)
    remove_watermark = np.clip(remove_watermark*4-140, 0, 255).astype(np.uint8)
    # 原图水印部分去除
    img_array[cor > 0] = 0
    # 原图与去水印部分叠加
    img_array = np.clip(img_array+remove_watermark, 0, 255).astype(np.uint8)
    return img_array


if __name__ == '__main__':
    input_dir = r'/home/hxzh02/文档/航拍数据集/'
    output_dir = r'/home/hxzh02/文档/航拍数据集/output'

    watermark_path = r'/home/hxzh02/文档/航拍数据集/m6240.jpg'
    img2 = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    visit_dir_files(input_dir, output_dir, input_dir, img2)
    print(u'完成！所有图片已保存至路径' + output_dir)