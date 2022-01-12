import cv2
import numpy as np

def Extend_line(x1, y1, x2, y2, x, y, flag):
    if flag == 1:
        if y1 == y2:
            return 0, y1, x, y2
        else:
            k = (y2 - y1) / (x2 - x1)
            b = (x1*y2-x2*y1)/(x1-x2)
            x3 = 0
            y3 = b
            x4 = x
            y4 = int(k * x4+b)
        return x3, y3, x4, y4
    else:
        if x1 == x2:
            return x1, 0, x2, y
        else:
            k = (y2 - y1) / (x2 - x1)
            b = (x1 * y2 - x2 * y1) / (x1 - x2)
            y3 = 0
            x3 = int(-1*b/k)
            y4 = y
            x4 = int((y4-b)/k)
            return x3, y3, x4, y4
