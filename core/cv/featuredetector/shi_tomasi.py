# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md

import numpy as np
import cv2 as cv

img = cv.imread("Lenna.png")
cv.namedWindow("Shi-Tomasi Corner Detection Test", cv.WINDOW_NORMAL)


def f(x=None):
    return


cv.createTrackbar("Max Corners", "Shi-Tomasi Corner Detection Test", 25, 100, f)
cv.createTrackbar("Threshold", "Shi-Tomasi Corner Detection Test", 39, 100, f)
cv.createTrackbar("Min Distance", "Shi-Tomasi Corner Detection Test", 7, 14, f)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

img_bak = img

while True:
    img = img_bak.copy()

    max_corners = cv.getTrackbarPos("Max Corners", "Shi-Tomasi Corner Detection Test")
    threshold = cv.getTrackbarPos("Threshold", "Shi-Tomasi Corner Detection Test") / 100
    min_distance = cv.getTrackbarPos("Min Distance", "Shi-Tomasi Corner Detection Test")

    if threshold <= 0:
        threshold = 0.001

    corners = cv.goodFeaturesToTrack(gray, max_corners, threshold, min_distance)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 2, (0, 255, 0), -1)

    cv.imshow("Shi-Tomasi Corner Detection Test", img)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
