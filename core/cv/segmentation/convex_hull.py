# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

# Load image
image = cv2.imread("convex_hull.png")

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Binarize the image
ret, thresh = cv2.threshold(gray, 176, 255, 0)

# Find contours
contours, hierarchy = cv2.findContours(
    thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
)

# Sort the contours and remove the largest frame contour
n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

# Iterate through each contour and draw convex hull
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
    cv2.imshow("Convex Hull", image)

cv2.waitKey(0)

cv2.destroyAllWindows()
