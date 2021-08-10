import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("BRISK", cv.WINDOW_NORMAL)


def f(x):
    return


# Initiate BRISK detector
cv.createTrackbar("Threshold", "BRISK", 30, 128, f)
cv.createTrackbar("Octaves", "BRISK", 3, 9, f)
cv.createTrackbar("Pattern Scale", "BRISK", 3, 9, f)

while True:
    current_threshold = cv.getTrackbarPos("Threshold", "BRISK")
    current_octaves = cv.getTrackbarPos("Octaves", "BRISK")
    current_scale = (cv.getTrackbarPos("Pattern Scale", "BRISK") + 1) / 3

    # Additional parameters for customization:
    # radiusList	defines the radii (in pixels) where the samples around a keypoint are taken (for keypoint scale 1).
    # numberList	defines the number of sampling points on the sampling circle. Must be the same size as radiusList..
    # dMax	threshold for the short pairings used for descriptor formation (in pixels for keypoint scale 1).
    # dMin	threshold for the long pairings used for orientation determination (in pixels for keypoint scale 1).
    # indexChange	index remapping of the bits.

    brisk = cv.BRISK_create(
        thresh=current_threshold, octaves=current_octaves, patternScale=current_scale
    )

    # find the keypoints with BRISK
    kp = brisk.detect(img, None)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

    cv.imshow("BRISK", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
