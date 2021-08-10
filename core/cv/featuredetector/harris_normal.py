# https://github.com/methylDragon/opencv-python-reference/blob/master/02%20OpenCV%20Feature%20Detection%20and%20Description.md

import numpy as np
import cv2 as cv

img = cv.imread("Lenna.png")
cv.namedWindow("Harris Corner Detection Test", cv.WINDOW_NORMAL)


def f(x=None):
    return


cv.createTrackbar("Harris Window Size", "Harris Corner Detection Test", 5, 25, f)
cv.createTrackbar("Harris Parameter", "Harris Corner Detection Test", 1, 100, f)
cv.createTrackbar("Sobel Aperture", "Harris Corner Detection Test", 1, 14, f)
cv.createTrackbar("Detection Threshold", "Harris Corner Detection Test", 1, 100, f)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

img_bak = img

while True:
    img = img_bak.copy()

    window_size = cv.getTrackbarPos(
        "Harris Window Size", "Harris Corner Detection Test"
    )
    harris_parameter = cv.getTrackbarPos(
        "Harris Parameter", "Harris Corner Detection Test"
    )
    sobel_aperture = cv.getTrackbarPos("Sobel Aperture", "Harris Corner Detection Test")
    threshold = cv.getTrackbarPos("Detection Threshold", "Harris Corner Detection Test")

    sobel_aperture = sobel_aperture * 2 + 1

    if window_size <= 0:
        window_size = 1

    dst = cv.cornerHarris(gray, window_size, sobel_aperture, harris_parameter / 100)

    # Result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > threshold / 100 * dst.max()] = [0, 0, 255]

    dst_show = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    dst_show = (255 * dst_show).astype(np.uint8)

    cv.imshow("Harris Corner Detection Test", np.hstack((img, dst_show)))

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
