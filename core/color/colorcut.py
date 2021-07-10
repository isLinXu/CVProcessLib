import cv2
import numpy as np

# Step1. 转换为HSV
img = cv2.imread('../../images/表计/test1.png')
hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Step2. 用颜色分割图像
low_range = np.array([0, 123, 100])
high_range = np.array([5, 255, 255])
th = cv2.inRange(hue_image, low_range, high_range)
cv2.imshow('hue_image', th)
cv2.waitKey()

# Step3. 形态学运算，膨胀
dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

# Step4. Hough Circle
circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=10, maxRadius=20)

# Step5. 绘制
if circles is not None:
    x, y, radius = circles[0][0]
    center = (x, y)
    cv2.circle(img, center, radius, (0, 255, 0), 2)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()