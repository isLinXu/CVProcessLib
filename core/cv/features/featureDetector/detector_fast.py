import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("FAST", cv.WINDOW_NORMAL)


def f(x):
    return


# Agast detector types
detector_types = {
    0: cv.FAST_FEATURE_DETECTOR_TYPE_5_8,
    1: cv.FAST_FEATURE_DETECTOR_TYPE_7_12,
    2: cv.FAST_FEATURE_DETECTOR_TYPE_9_16,
}

# Set Agast detector parameter callbacks
cv.createTrackbar("Threshold", "FAST", 15, 50, f)
cv.createTrackbar("Non Max Suppression", "FAST", 1, 1, f)
cv.createTrackbar("Type", "FAST", 0, len(detector_types) - 1, f)

while True:
    fast_threshold = cv.getTrackbarPos("Threshold", "FAST")
    non_max_suppression = cv.getTrackbarPos("Non Max Suppression", "FAST")
    current_type = cv.getTrackbarPos("Type", "FAST")

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create(
        threshold=fast_threshold,
        nonmaxSuppression=non_max_suppression,
        type=detector_types[current_type],
    )

    # find and draw the keypoints
    kp = fast.detect(img, None)

    # Flags:
    # cv.DRAW_MATCHES_FLAGS_DEFAULT,
    # cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    # cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    # cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    cv.imshow("FAST", img2)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
