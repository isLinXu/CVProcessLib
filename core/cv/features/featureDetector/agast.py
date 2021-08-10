import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("Agast", cv.WINDOW_NORMAL)


def f(x):
    return


# Agast detector types
detector_types = {
    0: cv.AGAST_FEATURE_DETECTOR_AGAST_5_8,
    1: cv.AGAST_FEATURE_DETECTOR_AGAST_7_12D,
    2: cv.AGAST_FEATURE_DETECTOR_AGAST_7_12S,
    3: cv.AGAST_FEATURE_DETECTOR_OAST_9_16,
}

# Set Agast detector parameter callbacks
cv.createTrackbar("Threshold", "Agast", 15, 50, f)
cv.createTrackbar("Non Max Suppression", "Agast", 1, 1, f)
cv.createTrackbar("Type", "Agast", 0, len(detector_types) - 1, f)

while True:
    agast_threshold = cv.getTrackbarPos("Threshold", "Agast")
    non_max_suppression = cv.getTrackbarPos("Non Max Suppression", "Agast")
    current_type = cv.getTrackbarPos("Type", "Agast")

    agast = cv.AgastFeatureDetector_create(
        threshold=agast_threshold,
        nonmaxSuppression=non_max_suppression,
        type=detector_types[current_type],
    )

    # find the keypoints with Agast
    kp = agast.detect(img, None)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

    cv.imshow("Agast", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
