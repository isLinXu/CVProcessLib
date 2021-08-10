# pip3 install opencv-contrib-python

import numpy as np
import cv2 as cv

img1 = cv.imread("box.png")  # queryImage
img2 = cv.imread("box_in_scene.png")  # trainImage

# Initiate BRISK detector
detector = cv.BRISK_create()

# Create Matcher window
cv.namedWindow("DAISY BF Matcher", cv.WINDOW_NORMAL)


def f(x):
    return


# BoostDesc descriptor types
norm_types = {
    0: 100,  # NRM_NONE
    1: 101,  # NRM_PARTIAL
    2: 102,  # NRM_FULL
    3: 103,  # NRM_SIFT
}

# Set DAISY descriptor parameter callbacks
cv.createTrackbar("Radius", "DAISY BF Matcher", 15, 50, f)
cv.createTrackbar("Q Radius", "DAISY BF Matcher", 2, 50, f)
cv.createTrackbar("Q Theta", "DAISY BF Matcher", 8, 50, f)
cv.createTrackbar("Q Hist", "DAISY BF Matcher", 8, 50, f)
cv.createTrackbar("Norm", "DAISY BF Matcher", 0, len(norm_types) - 1, f)
cv.createTrackbar("Interpolation", "DAISY BF Matcher", 1, 1, f)
cv.createTrackbar("Use Orientation", "DAISY BF Matcher", 0, 1, f)

while True:

    # find the keypoints with BRISK
    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)

    # Read matcher parameters
    current_radius = cv.getTrackbarPos("Radius", "DAISY BF Matcher")
    current_q_radius = cv.getTrackbarPos("Q Radius", "DAISY BF Matcher") + 1
    current_q_theta = cv.getTrackbarPos("Q Theta", "DAISY BF Matcher")
    current_q_hist = cv.getTrackbarPos("Q Hist", "DAISY BF Matcher")
    current_norm = cv.getTrackbarPos("Norm", "DAISY BF Matcher")
    current_interpolation = cv.getTrackbarPos("Interpolation", "DAISY BF Matcher")
    current_orientation = cv.getTrackbarPos("Use Orientation", "DAISY BF Matcher")

    # Initiate DAISY
    descriptor = cv.xfeatures2d.DAISY_create(
        radius=current_radius,
        q_radius=current_q_radius,
        q_theta=current_q_theta,
        q_hist=current_q_hist,
        norm=norm_types[current_norm],
        interpolation=current_interpolation,
        use_orientation=current_orientation,
    )

    # find the descriptors with DAISY
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2, des2 = descriptor.compute(img2, kp2)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

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
    cv.imshow("DAISY BF Matcher", img3)

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
