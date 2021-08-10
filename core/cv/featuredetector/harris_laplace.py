# pip3 install opencv-contrib-python

import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("HarrisLaplace", cv.WINDOW_NORMAL)


def f(x):
    return


# Set Agast detector parameter callbacks
cv.createTrackbar("Num Octaves", "HarrisLaplace", 5, 20, f)
cv.createTrackbar("Max Corners", "HarrisLaplace", 5, 100, f)
cv.createTrackbar("Corners Threshold", "HarrisLaplace", 1, 100, f)
cv.createTrackbar("DOG Threshold", "HarrisLaplace", 10, 100, f)
cv.createTrackbar("Num Layers", "HarrisLaplace", 1, 1, f)

while True:
    current_num_octaves = cv.getTrackbarPos("Num Octaves", "HarrisLaplace") + 1
    current_num_layers = 2 * (cv.getTrackbarPos("Num Layers", "HarrisLaplace") + 1)
    current_max_corners = cv.getTrackbarPos("Max Corners", "HarrisLaplace") * 100
    current_corners_thresh = (
        cv.getTrackbarPos("Corners Threshold", "HarrisLaplace") / 100
    )
    current_dog_thresh = cv.getTrackbarPos("DOG Threshold", "HarrisLaplace") / 1000

    # Initiate HarrisLaplace object with default values
    fast = cv.xfeatures2d.HarrisLaplaceFeatureDetector_create(
        numOctaves=current_num_octaves,
        maxCorners=current_max_corners,
        corn_thresh=current_corners_thresh,
        DOG_thresh=current_dog_thresh,
        num_layers=current_num_layers,
    )

    # find and draw the keypoints
    kp = fast.detect(img, None)

    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

    cv.imshow("HarrisLaplace", img2)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
