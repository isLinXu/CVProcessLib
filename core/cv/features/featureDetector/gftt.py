# Good Features To Track (GFTT)
import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("GFTT", cv.WINDOW_NORMAL)


def f(x):
    return


# Set GFTT detector parameter callbacks
cv.createTrackbar("Max Corners", "GFTT", 25, 100, f)
cv.createTrackbar("Quality Level", "GFTT", 1, 100, f)
cv.createTrackbar("Min Distance", "GFTT", 1, 100, f)
cv.createTrackbar("Block Size", "GFTT", 3, 10, f)
cv.createTrackbar("Gradiant Size", "GFTT", 3, 15, f)
cv.createTrackbar("Use Harris Detector", "GFTT", 1, 1, f)
cv.createTrackbar("k", "GFTT", 1, 100, f)

while True:
    max_corners = cv.getTrackbarPos("Max Corners", "GFTT")
    current_quality = (cv.getTrackbarPos("Quality Level", "GFTT") + 1) / 100
    current_distance = cv.getTrackbarPos("Min Distance", "GFTT") / 10
    use_harris = cv.getTrackbarPos("Use Harris Detector", "GFTT")
    block_size = cv.getTrackbarPos("Block Size", "GFTT") + 1
    gradiant_size = 2 * cv.getTrackbarPos("Gradiant Size", "GFTT") + 1
    current_k = cv.getTrackbarPos("k", "GFTT") / 400

    gftt = cv.GFTTDetector_create(
        maxCorners=max_corners,
        qualityLevel=current_quality,
        minDistance=current_distance,
        blockSize=block_size,
        gradiantSize=gradiant_size,
        useHarrisDetector=use_harris,
        k=current_k,
    )

    # find the keypoints with GFTT
    kp = gftt.detect(img, None)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

    cv.imshow("GFTT", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
