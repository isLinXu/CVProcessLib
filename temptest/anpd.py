# https://www.pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/

import numpy as np
import cv2


class ANPD:
    def __init__(self, minAR=4, maxAR=5):
        self.minAR = minAR
        self.maxAR = maxAR

    def imshow(self, title, image):
        cv2.imshow(title, image)
        cv2.waitKey(0)

    def locate_licence_plate_candidates(self, gray, keep=5):
        # Blackhat morphological operation to reveal dark regions on light backgrounds
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.imshow("Blackhat", blackHat)

        # Find regions in image that are light
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.imshow("Light Regions", light)

        # Scharr gradient representation of the blackhat image
        # x direction , rescaled in [0..255]
        gradX = cv2.Sobel(blackHat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        minVal, maxVal = np.min(gradX), np.max(gradX)
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.imshow("Scharr", gradX)

        # Blur the gradient representation, apply close operation,
        # threshold the image using Otsu's method
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.imshow("Grad Thresh", thresh)

        # Perform a series of erosions and dilations to clean up the thresholded image
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.imshow("Grad Erode/Dilate", thresh)

        # Take the bitwise AND between the threshold result and
        # the light regions of the image
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.imshow("Final", thresh)

        # Find contours in the thresholded image and sort them
        # by size in descending order
        cnts, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        return cnts

    def locate_license_plate(self, gray, candidates):
        # Initialize license plate contour and ROI
        lpCnt = None
        roi = None

        # Loop over license plate candidate contours
        for c in candidates:
            # Compute the bounding box of the contour
            # and use it to derive aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # Check to see if the aspect ratio is rectangular
            if ar >= self.minAR and ar <= self.maxAR:
                # Store the contour and extract the license plate
                # Then threshold it
                lpCnt = c
                licensePlate = gray[y : y + h, x : x + w]
                roi = cv2.threshold(
                    licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )[1]

                self.imshow("License Plate", licensePlate)
                self.imshow("ROI", roi)
                break

        return roi, lpCnt


anpd = ANPD()

gray = cv2.imread("anpd_input.jpg", cv2.IMREAD_GRAYSCALE)

candidates = anpd.locate_licence_plate_candidates(gray)
roi, lpCnt = anpd.locate_license_plate(gray, candidates)

# Bounding rectangle of the license plate
print(cv2.boundingRect(lpCnt))
