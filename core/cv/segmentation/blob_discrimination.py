# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

# Image loaded as grayscale three channel
image = cv2.imread("blob_discrimination.png", cv2.IMREAD_GRAYSCALE)
# cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
cv2.imshow("Input Image", image)
cv2.waitKey(0)

# Set detector with default parameters
detector = cv2.SimpleBlobDetector_create()

# Detect blobs
keypoints = detector.detect(image)

# Draw detected blobs as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(
    image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Print blobs on image
number_of_blobs = len(keypoints)
text = "total number of blobs: " + str(number_of_blobs)
cv2.putText(blobs, text, (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

# Show keypoints
cv2.imshow("Blobs with default parameters", blobs)
cv2.waitKey(0)

# Initialize parameters for simple blob detection
params = cv2.SimpleBlobDetector_Params()

# Set area filtering parameters
params.filterByArea = True
params.minArea = 100

# Set circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.8

# Set convexity filtering parameter
params.filterByConvexity = True
params.minConvexity = 0.2

# Set inertia filtering parameter
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Set detector with defined parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw detected blobs as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(
    image, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Print blobs on image
number_of_blobs = len(keypoints)
text = "total number of circular blobs: " + str(number_of_blobs)
cv2.putText(blobs, text, (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

# Show keypoints
cv2.imshow("Circular Blobs", blobs)
cv2.waitKey(0)

cv2.destroyAllWindows()
