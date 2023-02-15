# -*- coding: utf-8 -*-
"""
Created on Thu Nov 1 10:43:29 2018
@author: Administrator
"""
import os
import cv2
import numpy as np

path = './1_900_0_1'
file_path = './1_900_0_1/1.jpg'

def compute(path):
    file_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        # img = cv2.imread(os.path.join(path, file_name), 1)
        img = cv2.imdecode(np.fromfile(os.path.join(path, file_name), dtype=np.uint8), 1)
        per_image_Bmean.append(np.mean(img[:, :, 0]))
        per_image_Gmean.append(np.mean(img[:, :, 1]))
        per_image_Rmean.append(np.mean(img[:, :, 2]))
        print("b："+str(np.mean(img[:, :, 0])))
        print("g："+str(np.mean(img[:, :, 1])))
        print("r："+str(np.mean(img[:, :, 2])))
        print(np.mean(img[:, :, 2])-np.mean(img[:, :, 1]))
        cv2.imshow("1",img)
        cv2.waitKey(0)
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return R_mean, G_mean, B_mean


def compute_one(img):
    R_mean = np.mean(img[:, :, 0])
    G_mean = np.mean(img[:, :, 1])
    B_mean = np.mean(img[:, :, 2])
    return R_mean, G_mean, B_mean

if __name__ == '__main__':
    # R, G, B = compute(path)
    # print(R, G, B)
    img = cv2.imread(file_path)
    cv2.imshow('img', img)
    cv2.waitKey()
    r,g,b = compute_one(img)
    print('r,g,b:', r,g,b)