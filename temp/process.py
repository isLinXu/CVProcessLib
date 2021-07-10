# -*- coding: utf-8 -*-
import cv2
import numpy as np

if __name__ == '__main__':
    file = '/home/linxu/Desktop/1_0_0_1020_1_0/状态指示器/Green/250.jpeg'
    img = cv2.imread(file)
    rows, cols, ch = img.shape
    SIZE = 3  # 卷积核大小
    P = int(SIZE / 2)
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    BEGIN = False
    BP = []

    for row in range(P, rows - P, 1):
        for col in range(P, cols - P, 1):
            # print(img[row,col])
            if (img[row, col] == WHITE).all():
                kernal = []
                for i in range(row - P, row + P + 1, 1):
                    for j in range(col - P, col + P + 1, 1):
                        kernal.append(img[i, j])
                        if (img[i, j] == BLACK).all():
                            print(i,j)
                            BP.append([i, j])

    print(len(BP))
    uniqueBP = np.array(list(set([tuple(c) for c in BP])))
    print(len(uniqueBP))

    for x, y in uniqueBP:
        img[x, y] = WHITE

    cv2.imshow('img', img)
    cv2.imwrite('second_bird.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    # img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # print(img2gray.shape)
    # mask = cv2.bitwise_and(img,img,mask=img2gray)
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()