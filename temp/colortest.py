
import cv2
import numpy as np


if __name__ == '__main__':
    file = '/home/linxu/Desktop/1_0_0_1020_1_0/109832_steerPoint53_preset806_20210509090051_v.jpeg'
    src = cv2.imread(file)
    cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input", src)
    """
    提取图中的红色部分
    """
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    # low_hsv = np.array([0, 43, 46])
    # high_hsv = np.array([10, 255, 255])
    low_hsv = np.array([35, 77, 43])
    high_hsv = np.array([255, 46, 255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    cv2.imshow("test", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()