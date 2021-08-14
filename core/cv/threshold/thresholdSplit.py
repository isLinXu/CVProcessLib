import numpy as np
import cv2

'''
Paper Title: A threshold selection method from gray-level histograms
Ostu二值分割阈值选取算法
'''
def get_Otsu_value(delta):
    val = np.zeros([256])
    for th in range(256):
        loc1 = delta > th
        loc2 = delta <= th

        if delta[loc1].size == 0:
            mu1 = 0
            omega1 = 0
        else:
            mu1 = np.mean(delta[loc1])
            omega1 = delta[loc1].size / delta.size

        if delta[loc2].size==0:
            mu2 = 0
            omega2 = 0
        else:
            mu2 = np.mean(delta[loc2])
            omega2 = delta[loc2].size / delta.size
        val[th] = omega1*omega2*np.power((mu1 -mu2), 2)

    loc = np.where(val ==np.max(val))
    print("The best Otsu threshold is: ", loc[0])
    return loc[0]

if __name__ == '__main__':
    file = '/home/linxu/PycharmProjects/CVProcessLib/images/表计/1.jpeg'
    img = cv2.imread(file)
    cv2.imshow('img', img)
    dst = get_Otsu_value(img)
    print(dst)
    # cv2.imshow('dst', dst)
    cv2.waitKey()
