# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

# Load shape template
template_image = cv2.imread("matching_contour_template.png")
template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow("Template", cv2.WINDOW_NORMAL)
cv2.imshow("Template", template)
cv2.waitKey(0)

# Load the image to match the template on
target = cv2.imread("matching_contour.png")
gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Threshold the images
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(gray, 127, 255, 0)

# Find contours in template
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Sort the contours and extract the largest largest contour
# This will be the template contour
sorted(contours, key=cv2.contourArea, reverse=False)
template_contour = contours[1]

# Extract the contours from the target image
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Iterate through each contour to compare shape
# and to find the best match
closest_contour = []
best_match = 0.16
for c in contours:
    match = cv2.matchShapes(template_contour, c, 1, 0.0)
    print(match)
    if match < best_match:
        closest_contour = c
        best_match = match

cv2.drawContours(target, [closest_contour], -1, (0, 255, 0), 3)
cv2.imshow("Output", target)

cv2.waitKey(0)

cv2.destroyAllWindows()
