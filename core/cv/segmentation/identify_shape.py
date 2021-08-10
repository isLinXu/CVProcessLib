# https://circuitdigest.com/tutorial/image-segmentation-using-opencv

import cv2
import numpy as np

# Load the image to identify shapes
image = cv2.imread("identify_shape.png")

# cv2.namedWindow("Identify Shapes", cv2.WINDOW_NORMAL)
cv2.imshow("Identify Shapes", image)
cv2.waitKey(0)

# Grayscale and binarize
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Sort the contours and remove the largest frame contour
n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

# Get approximate polygons
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

    if len(approx) == 3:
        shape_name = "Triangle"
        cv2.drawContours(image, [cnt], 0, (0, 255, 0), -1)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(
            image, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1
        )
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # cv2.boundingRect return the left width and height in pixels, starting from the top
        # left corner, for square it would be roughly same
        if abs(w - h) <= 3:
            shape_name = "square"
            # find contour center to place text at center
            cv2.drawContours(image, [cnt], 0, (0, 125, 255), -1)
            cv2.putText(
                image,
                shape_name,
                (cx - 50, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                1,
            )
        else:
            shape_name = "Rectangle"
            # find contour center to place text at center
            cv2.drawContours(image, [cnt], 0, (0, 0, 255), -1)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                image,
                shape_name,
                (cx - 50, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                1,
            )
    elif len(approx) == 10:
        shape_name = "star"
        cv2.drawContours(image, [cnt], 0, (255, 255, 0), -1)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(
            image, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1
        )
    elif len(approx) >= 15:
        shape_name = "circle"
        cv2.drawContours(image, [cnt], 0, (0, 255, 255), -1)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(
            image, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1
        )

    cv2.imshow("identifying shapes", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
