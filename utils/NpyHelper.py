# -*- coding:utf-8 -*-

"""
@ Author: LinXu
@ Contact: 17746071609@163.com
@ Date: 2022/01/15 15:35 PM
@ Software: PyCharm
@ File: NpyHelper.py
@ Desc: .npy格式相关支持函数
"""

import os
import numpy as np

def readFiledir_saveNpy(file_dir, npy_name):
    a = os.listdir(file_dir)  # 读取文件夹中的目录文件
    print(a)
    save_path = npy_name + '.npy'
    np.save(save_path, a)
    return save_path


def read_npyfile(npy_path):
    # 读取.npy文件
    arr = np.load(npy_path)
    print(arr)
    return arr

if __name__ == '__main__':
    # 文件夹的路径
    # file_dir = '/media/hxzh02/SB@home/hxzh/Dataset/Plane_detect_datasets/VOCdevkit_lineextract_detect/VOC2007/Annotations/'
    file_dir = '/media/hxzh02/SB@home/hxzh/Dataset/Plane_detect_datasets/VOCdevkit_windbias_detect/VOC2007/Annotations/'
    npy_name = 'xml'
    readFiledir_saveNpy(file_dir,npy_name)

    # npy_path = 'xml.npy'
    # read_npyfile(npy_path)