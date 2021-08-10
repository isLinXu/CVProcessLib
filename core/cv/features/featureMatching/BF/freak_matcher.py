# pip3 install opencv-contrib-python

import numpy as np
import cv2 as cv

img1 = cv.imread("box.png")  # queryImage
img2 = cv.imread("box_in_scene.png")  # trainImage

# Initiate BRISK detector
detector = cv.BRISK_create()

# Create Matcher window
cv.namedWindow("FREAK BF Matcher", cv.WINDOW_NORMAL)


def f(x):
    return


# Set FREAK descriptor parameter callbacks
cv.createTrackbar("Normalize Orientation", "FREAK BF Matcher", 1, 1, f)
cv.createTrackbar("Normalize Scale", "FREAK BF Matcher", 1, 1, f)
cv.createTrackbar("Pattern Scale", "FREAK BF Matcher", 22, 50, f)
cv.createTrackbar("Octaves", "FREAK BF Matcher", 4, 16, f)

while True:

    # find the keypoints with BRISK
    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)

    # Read mactcher parameters
    current_norm_orientation = cv.getTrackbarPos(
        "Normalize Orientation", "FREAK BF Matcher"
    )
    current_norm_scale = cv.getTrackbarPos("Normalize Scale", "FREAK BF Matcher")
    current_pattern_scale = cv.getTrackbarPos("Pattern Scale", "FREAK BF Matcher")
    current_octaves = cv.getTrackbarPos("Octaves", "FREAK BF Matcher")

    # Initiate FREAK
    descriptor = cv.xfeatures2d.FREAK_create(
        orientationNormalized=current_norm_orientation,
        scaleNormalized=current_norm_scale,
        patternScale=current_pattern_scale,
        nOctaves=current_octaves,
    )

    # find the descriptors with FREAK
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
    cv.imshow("FREAK BF Matcher", img3)

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
