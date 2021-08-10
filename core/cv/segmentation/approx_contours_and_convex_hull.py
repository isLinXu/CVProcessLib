# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

# Load the image and keep a copy
image = cv2.imread("approx_contours_and_convex_hull.png")
orig_img = image.copy()

# cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
cv2.imshow("Input Image", image)
cv2.waitKey(0)

# Grayscale an binarize the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(
    thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
)

# Iterate through each contour and compute bounding rectangle
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Bounding Rect", orig_img)

cv2.waitKey(0)

# Iterate through each contour and compute the approx contour
# Calculate accuracy as a percent of contour perimeter
for c in contours:
    # True stands for closed curve in both cases
    accuracy = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.imshow("Approx PolyDP", image)

cv2.waitKey(0)

cv2.destroyAllWindows()
