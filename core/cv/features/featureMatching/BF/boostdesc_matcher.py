# pip3 install opencv-contrib-python

import numpy as np
import cv2 as cv

img1 = cv.imread("/home/linxu/PycharmProjects/CVProcessLib/core/cv/features/featureMatching/box.png")  # queryImage
img2 = cv.imread("/home/linxu/PycharmProjects/CVProcessLib/core/cv/features/featureMatching/box_in_scene.png")  # trainImage

# Initiate BRISK detector
detector = cv.BRISK_create()

# Create Matcher window
cv.namedWindow("BoostDesc BF Matcher", cv.WINDOW_NORMAL)


def f(x):
    return


# BoostDesc descriptor types
matcher_types = {
    0: 100,  # BGM
    1: 101,  # BGM_HARD
    2: 102,  # BGM_BILINEAR
    3: 300,  # BINBOOST_64
    4: 301,  # BINBOOST_128
    5: 302,  # BINBOOST_256
    6: 200,  # LBGM,
}

# Set BoostDesc descriptor parameter callbacks
cv.createTrackbar("Type", "BoostDesc BF Matcher", 5, len(matcher_types) - 1, f)
cv.createTrackbar("Scale Orientation", "BoostDesc BF Matcher", 1, 1, f)
cv.createTrackbar("Scale Factor", "BoostDesc BF Matcher", 25, 50, f)

while True:

    # find the keypoints with BRISK
    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)

    # Read mactcher parameters
    current_type = cv.getTrackbarPos("Type", "BoostDesc BF Matcher")
    current_scale_orientation = cv.getTrackbarPos(
        "Scale Orientation", "BoostDesc BF Matcher"
    )

    # 6.25f is default and fits for KAZE, SURF detected keypoints
    # 6.75f should be the scale for SIFT detected keypoints
    # 5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK
    # 0.75f should be the scale for ORB keypoints
    # 1.50f was the default in original implementation
    current_scale_factor = (
        cv.getTrackbarPos("Scale Factor", "BoostDesc BF Matcher") * 0.25
    )

    # Initiate BoostDesc
    descriptor = cv.xfeatures2d.BoostDesc_create(
        desc=matcher_types[current_type],
        use_scale_orientation=current_scale_orientation,
        scale_factor=current_scale_factor,
    )

    # find the descriptors with BoostDesc
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2, des2 = descriptor.compute(img2, kp2)

    # create BFMatcher object
    if current_type != 6:  # not LBGM
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    else:
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
    cv.imshow("BoostDesc BF Matcher", img3)

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
