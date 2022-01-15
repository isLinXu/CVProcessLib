
import os
import numpy as np

def readFiledir_saveNpy(file_dir, npy_name):
    a = os.listdir(file_dir)  # 读取文件夹中的目录文件
    print(a)
    save_path = npy_name + '.npy'
    np.save(save_path, a)


def read_npyfile(npy_path):
    # 读取.npy文件
    arr = np.load(npy_path)
    print(arr)

if __name__ == '__main__':
    file_dir = '/media/hxzh02/SB@home/hxzh/Dataset/Plane_detect_datasets/VOCdevkit_lineextract_detect/VOC2007/Annotations/'  # 文件夹的路径
    npy_name = 'xml'
    readFiledir_saveNpy(file_dir,npy_name)

    npy_path = 'xml.npy'
    read_npyfile(npy_path)