# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

image = cv2.imread("segmentation_and_contours.png")
# cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
cv2.imshow("Input Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edged = cv2.Canny(gray, 30, 200)
# cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow("Canny Edges", edged)
cv2.waitKey(0)

# Find Contours
# Use a copy of image as finding contours alters the image
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.namedWindow("Canny Edges after contouring", cv2.WINDOW_NORMAL)
cv2.imshow("Canny Edges after contouring", edged)
cv2.waitKey(0)

print(contours)
print(f"Number of contours found = {len(contours)}")

cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
# cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
cv2.imshow("Contours", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
