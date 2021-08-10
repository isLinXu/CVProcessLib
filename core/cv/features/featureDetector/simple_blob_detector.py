import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("SimpleBlobDetector", cv.WINDOW_NORMAL)


def f(x):
    return


# Set SimpleBlobDetector detector parameter callbacks
cv.createTrackbar("Min Threshold", "SimpleBlobDetector", 0, 255, f)
cv.createTrackbar("Max Threshold", "SimpleBlobDetector", 255, 255, f)
cv.createTrackbar("Threshold Step", "SimpleBlobDetector", 0, 30, f)
cv.createTrackbar("Min Dist Between Blobs", "SimpleBlobDetector", 0, 30, f)
cv.createTrackbar("Min Repeatability", "SimpleBlobDetector", 0, 30, f)
cv.createTrackbar("Blob Color", "SimpleBlobDetector", 0, 1, f)
cv.createTrackbar("Min Area", "SimpleBlobDetector", 5, 20, f)
cv.createTrackbar("Max Area", "SimpleBlobDetector", 15, 20, f)
cv.createTrackbar("Min Circularity", "SimpleBlobDetector", 5, 20, f)
cv.createTrackbar("Max Circularity", "SimpleBlobDetector", 15, 20, f)
cv.createTrackbar("Min Inertia", "SimpleBlobDetector", 5, 20, f)
cv.createTrackbar("Max Inertia", "SimpleBlobDetector", 15, 20, f)
cv.createTrackbar("Min Convexity", "SimpleBlobDetector", 5, 20, f)
cv.createTrackbar("Max Convexity", "SimpleBlobDetector", 15, 20, f)

while True:
    current_min_threshold = cv.getTrackbarPos("Min Threshold", "SimpleBlobDetector")
    current_max_threshold = cv.getTrackbarPos("Max Threshold", "SimpleBlobDetector")
    current_threshold_step = (
        cv.getTrackbarPos("Threshold Step", "SimpleBlobDetector") + 10
    )
    current_min_blob_dist = cv.getTrackbarPos(
        "Min Dist Between Blobs", "SimpleBlobDetector"
    )
    current_min_repeatability = (
        cv.getTrackbarPos("Min Repeatability", "SimpleBlobDetector") + 2
    )
    current_blob_color = cv.getTrackbarPos("Blob Color", "SimpleBlobDetector") * 255
    current_min_area = cv.getTrackbarPos("Min Area", "SimpleBlobDetector")
    current_max_area = cv.getTrackbarPos("Max Area", "SimpleBlobDetector")
    current_min_circularity = (
        cv.getTrackbarPos("Min Circularity", "SimpleBlobDetector") / 10
    )
    current_max_circularity = (
        cv.getTrackbarPos("Max Circularity", "SimpleBlobDetector") / 10
    )
    current_min_inertia = cv.getTrackbarPos("Min Inertia", "SimpleBlobDetector") / 10
    current_max_inertia = cv.getTrackbarPos("Max Inertia", "SimpleBlobDetector") / 10
    current_min_convexity = (
        cv.getTrackbarPos("Min Convexity", "SimpleBlobDetector") / 10
    )
    current_max_convexity = (
        cv.getTrackbarPos("Max Convexity", "SimpleBlobDetector") / 10
    )

    params = cv.SimpleBlobDetector_Params()

    params.minDistBetweenBlobs = current_min_blob_dist
    params.minRepeatability = current_min_repeatability

    # Convert the source image to binary images by applying thresholding with several thresholds
    # from minThreshold (inclusive) to maxThreshold (exclusive) with
    # distance thresholdStep between neighboring thresholds.
    params.minThreshold = current_min_threshold
    params.maxThreshold = current_max_threshold
    params.thresholdStep = current_threshold_step

    # This filter compares the intensity of a binary image at the center of a blob to blobColor.
    # If they differ, the blob is filtered out.
    params.filterByColor = True
    params.blobColor = current_blob_color

    # Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
    params.filterByArea = True
    params.minArea = current_min_area
    params.maxArea = current_max_area

    # Extracted blobs have circularity ( 4∗π∗Areaperimeter∗perimeter)
    # between minCircularity (inclusive) and maxCircularity (exclusive).
    params.filterByCircularity = True
    params.minCircularity = current_min_circularity
    params.maxCircularity = current_max_circularity

    # Extracted blobs have this ratio between minInertiaRatio (inclusive) and maxInertiaRatio (exclusive).
    params.filterByInertia = True
    params.minInertiaRatio = current_min_inertia
    params.maxInertiaRatio = current_max_inertia

    # Extracted blobs have convexity (area / area of blob convex hull)
    # between minConvexity (inclusive) and maxConvexity (exclusive).
    params.filterByConvexity = True
    params.minConvexity = current_min_convexity
    params.maxConvexity = current_max_convexity

    blob_detector = cv.SimpleBlobDetector_create(params)

    # find the keypoints with SimpleBlobDetector
    kp = blob_detector.detect(img, None)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0))

    cv.imshow("SimpleBlobDetector", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
