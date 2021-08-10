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

    # Threshold for an optimal value, it may vary depending on the image.
    _, dst_thresh = cv.threshold(dst, threshold / 100 * dst.max(), 255, 0)
    dst_thresh = np.uint8(dst_thresh)

    dst_show = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    dst_show = np.uint8(dst_show)

    # REFINE CORNERS HERE!

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst_thresh)

    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    try:
        # Now draw them
        corners = np.int0(corners)

        img[corners[:, 1], corners[:, 0]] = [0, 255, 0]
        img[dst_thresh > 1] = [0, 0, 255]

    except:
        pass

    cv.imshow("Harris Corner Detection Test", np.hstack((img, dst_show)))

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
