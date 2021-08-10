# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

# Image loaded as grayscale three channel
image = cv2.imread("blob_detection.jpg", cv2.IMREAD_GRAYSCALE)
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
    image, keypoints, blank, (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT
)

# Show keypoints
cv2.imshow("Blobs", blobs)

cv2.waitKey(0)

cv2.destroyAllWindows()
