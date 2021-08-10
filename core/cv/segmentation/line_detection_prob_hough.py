# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

image = cv2.imread("line_detection.png")
# cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
cv2.imshow("Input Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)

# Run Hough lines using rho accuracy of 1 pixel
# Theta accuracy of (np.pi/180) which is 1 degree
# Minimum vote (points along the line) of 100
# Min line length 5 pixels
# Max gap between lines 10 pixels
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 100, 10)

# Iterate through each line to convert if for printing
for i in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Probabilistic Hough Lines", image)

cv2.waitKey(0)

cv2.destroyAllWindows()
