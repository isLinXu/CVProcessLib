# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

image = cv2.imread("line_detection.png")
# cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
cv2.imshow("Input Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 170, apertureSize=3)
# cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)

# Run Hough lines using rho accuracy of 1 pixel
# Theta accuracy of (np.pi/180) which is 1 degree
# Line threshold is set to 150 (number of points per line)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

# Iterate through each line to convert if for printing
for i in range(0, len(lines)):
    for (
        rho,
        theta,
    ) in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Hough Lines", image)

cv2.waitKey(0)

cv2.destroyAllWindows()
