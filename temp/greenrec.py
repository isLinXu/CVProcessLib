
'''
颜色过滤:绿色区域框选

'''

if __name__ == '__main__':
    # coding=utf-8
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    path = '/home/linxu/Documents/Work/sources/状态指示器图片/16.png'
    img = cv2.imread(path)
    b = img[:, :, 0].astype("int16")
    g = img[:, :, 1].astype("int16")
    r = img[:, :, 2].astype("int16")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    decision = np.int8(g>b+20)+np.int8(g>r+20)
    # decision = np.int8(g > b + 35) + np.int8(g > r + 35)
    mask[decision == 2] = 255
    cv2.imshow('mask', mask)
    cv2.waitKey()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    areas = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        points.append([x, y, w, h])
        areas.append(w * h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
    # areas = [point[2] * point[3] for point in points]
    areas = np.array(areas)
    points = np.array(points)
    print(areas)
    area = areas[np.argmax(areas)]
    point = points[np.argmax(areas)]
    x, y, w, h = point
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imshow('img', img)
    cv2.waitKey()
    print(area)
    print(point)