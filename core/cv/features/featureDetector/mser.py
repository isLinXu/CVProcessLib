import cv2 as cv

img = cv.imread("Lenna.png", flags=cv.IMREAD_GRAYSCALE)

cv.namedWindow("MSER", cv.WINDOW_NORMAL)


def f(x):
    return


# Set MSER detector parameter callbacks
cv.createTrackbar("Delta", "MSER", 3, 20, f)
cv.createTrackbar("Min Area", "MSER", 0, 20, f)
cv.createTrackbar("Max Area", "MSER", 0, 20, f)
cv.createTrackbar("Max Variation", "MSER", 0, 20, f)

while True:
    current_delta = cv.getTrackbarPos("Delta", "MSER")
    current_min_area = cv.getTrackbarPos("Min Area", "MSER") + 5
    current_max_area = cv.getTrackbarPos("Max Area", "MSER") + 5
    current_max_variation = cv.getTrackbarPos("Max Variation", "MSER") / 10

    mser = cv.MSER_create(
        _delta=current_delta,
        _min_area=current_min_area,
        _max_area=current_max_area,
        _max_variation=current_max_variation,
    )

    # find the keypoints with MSER
    kp = mser.detect(img, None)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

    cv.imshow("MSER", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
