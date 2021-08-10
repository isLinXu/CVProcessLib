# pip3 install opencv-contrib-python

import numpy as np
import cv2 as cv

img1 = cv.imread("box.png")  # queryImage
img2 = cv.imread("box_in_scene.png")  # trainImage

# Initiate BRISK detector
detector = cv.BRISK_create()

# Create Matcher window
cv.namedWindow("LUCID BF Matcher", cv.WINDOW_NORMAL)


def f(x):
    return


# Set BRIEF descriptor parameter callbacks
cv.createTrackbar("Lucid Kernel", "LUCID BF Matcher", 1, 10, f)
cv.createTrackbar("Blur Kernel", "LUCID BF Matcher", 2, 10, f)

while True:

    # find the keypoints with BRISK
    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)

    # Read matcher parameters
    current_lucid_kernel = cv.getTrackbarPos("Lucid Kernel", "LUCID BF Matcher")
    current_blur_kernel = cv.getTrackbarPos("Blur Kernel", "LUCID BF Matcher")

    # Initiate LUCID
    descriptor = cv.xfeatures2d.LUCID_create(
        lucid_kernel=current_lucid_kernel, blur_kernel=current_blur_kernel
    )

    # find the descriptors with LUCID
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2, des2 = descriptor.compute(img2, kp2)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Flags:
    # cv.DRAW_MATCHES_FLAGS_DEFAULT
    # cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    # cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    # cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    img3 = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
    )

    # Draw matches
    cv.imshow("LUCID BF Matcher", img3)

    # Calculate homography
    # Consider point filtering
    obj = []
    scene = []
    for match in matches:
        obj.append(kp1[match.queryIdx].pt)
        scene.append(kp2[match.trainIdx].pt)

    # Calculate homography: Inliers and outliers
    # RANSAC, LMEDS, RHO
    H, _ = cv.findHomography(np.array(obj), np.array(scene), cv.RANSAC)

    if H is not None:
        # Frame of the object image
        obj_points = np.array(
            [
                [0, 0],
                [img1.shape[1], 0],
                [img1.shape[1], img1.shape[0]],
                [0, img1.shape[0]],
            ],
            dtype=np.float,
        )

        # Check the sanity of the transformation
        warped_points = cv.perspectiveTransform(np.array([obj_points]), H)

        warped_image = np.copy(img2)
        cv.drawContours(
            warped_image, np.array([warped_points]).astype(np.int32), 0, (0, 0, 255)
        )

        cv.namedWindow("Warped Object", cv.WINDOW_NORMAL)
        cv.imshow("Warped Object", warped_image)
    else:
        print("Error calculating perspective transformation")

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
