# pip3 install opencv-contrib-python

import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("STAR", cv.WINDOW_NORMAL)


def f(x):
    return


# Set Agast detector parameter callbacks
cv.createTrackbar("Max Size", "STAR", 9, 20, f)
cv.createTrackbar("ResponseThreshold", "STAR", 30, 100, f)
cv.createTrackbar("Line Threshold Projected", "STAR", 10, 30, f)
cv.createTrackbar("Line Threshold Binarized", "STAR", 8, 20, f)
cv.createTrackbar("Suppress Nonmax Size", "STAR", 9, 20, f)

while True:
    current_max_size = cv.getTrackbarPos("Max Size", "STAR") * 5
    current_resp_thresh = cv.getTrackbarPos("ResponseThreshold", "STAR")
    current_line_thresh_proj = cv.getTrackbarPos("Line Threshold Projected", "STAR")
    current_line_thresh_bin = cv.getTrackbarPos("Line Threshold Binarized", "STAR")
    current_suppress_nonmax_size = cv.getTrackbarPos("Suppress Nonmax Size", "STAR")

    # Initiate STAR detector
    star = cv.xfeatures2d.StarDetector_create(
        maxSize=current_max_size,
        responseThreshold=current_resp_thresh,
        lineThresholdProjected=current_line_thresh_proj,
        lineThresholdBinarized=current_line_thresh_bin,
        suppressNonmaxSize=current_suppress_nonmax_size,
    )

    # find the keypoints with STAR
    kp = star.detect(img, None)

    img_kp = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    cv.imshow("STAR", img_kp)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
