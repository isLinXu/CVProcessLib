import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("MSER", cv.WINDOW_NORMAL)


def f(x):
    return


# Set MSER detector parameter callbacks
cv.createTrackbar("Delta", "MSER", 5, 20, f)
cv.createTrackbar("Min Area", "MSER", 11, 30, f)
cv.createTrackbar("Max Area", "MSER", 3, 20, f)
cv.createTrackbar("Max Variation", "MSER", 10, 20, f)
cv.createTrackbar("Min Diversity", "MSER", 10, 20, f)
cv.createTrackbar("Max Evolution", "MSER", 10, 40, f)
cv.createTrackbar("Area Threshold", "MSER", 10, 20, f)
cv.createTrackbar("Min Margin", "MSER", 3, 30, f)
cv.createTrackbar("Edge Blur Size", "MSER", 2, 10, f)

while True:
    current_delta = cv.getTrackbarPos("Delta", "MSER")
    current_min_area = 5 * cv.getTrackbarPos("Min Area", "MSER") + 5
    current_max_area = 500 * cv.getTrackbarPos("Max Area", "MSER") + 5
    current_max_variation = cv.getTrackbarPos("Max Variation", "MSER") / 40
    # Color image related parameters
    current_min_diversity = cv.getTrackbarPos("Min Diversity", "MSER") / 10
    current_max_evolution = cv.getTrackbarPos("Max Evolution", "MSER") * 10
    current_area_threshold = cv.getTrackbarPos("Area Threshold", "MSER") / 5
    current_min_margin = cv.getTrackbarPos("Min Margin", "MSER") / 100
    current_edge_blur_size = 2 * cv.getTrackbarPos("Edge Blur Size", "MSER") + 1

    mser = cv.MSER_create(
        _delta=current_delta,
        _min_area=current_min_area,
        _max_area=current_max_area,
        _max_variation=current_max_variation,
        _min_diversity=current_min_diversity,
        _max_evolution=current_max_evolution,
        _area_threshold=current_area_threshold,
        _min_margin=current_min_margin,
        _edge_blur_size=current_edge_blur_size,
    )

    # find the keypoints with MSER
    kp = mser.detect(img, None)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

    cv.imshow("MSER", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
