import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("AKAZE", cv.WINDOW_NORMAL)


def f(x):
    return


descriptor_types = {
    0: cv.AKAZE_DESCRIPTOR_KAZE,
    1: cv.AKAZE_DESCRIPTOR_KAZE_UPRIGHT,
    2: cv.AKAZE_DESCRIPTOR_MLDB,
    3: cv.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,
}

diffusivity_types = {
    0: cv.KAZE_DIFF_PM_G1,
    1: cv.KAZE_DIFF_PM_G2,
    2: cv.KAZE_DIFF_WEICKERT,
    3: cv.KAZE_DIFF_CHARBONNIER,
}

# Initiate AKAZE detector
cv.createTrackbar("Descriptor Type", "AKAZE", 0, len(descriptor_types) - 1, f)
cv.createTrackbar("Diffusivity Type", "AKAZE", 0, len(diffusivity_types) - 1, f)
cv.createTrackbar("Threshold", "AKAZE", 10, 100, f)
cv.createTrackbar("Octaves", "AKAZE", 4, 16, f)
cv.createTrackbar("Octave Layers", "AKAZE", 4, 16, f)

while True:
    current_descriptor_type = cv.getTrackbarPos("Descriptor Type", "AKAZE")
    current_diffusivity_type = cv.getTrackbarPos("Diffusivity Type", "AKAZE")
    current_threshold = cv.getTrackbarPos("Threshold", "AKAZE") / 10000
    current_octaves = cv.getTrackbarPos("Octaves", "AKAZE") + 1
    current_octave_layers = cv.getTrackbarPos("Octave Layers", "AKAZE") + 1

    akaze = cv.AKAZE_create(
        descriptor_type=descriptor_types[current_descriptor_type],
        # descriptor_size does not influence the feature detection
        threshold=current_threshold,
        # descriptor_channels  does not influence the feature detection - up to 3
        nOctaves=current_octaves,
        nOctaveLayers=current_octave_layers,
        diffusivity=diffusivity_types[current_diffusivity_type],
    )

    # find the keypoints with AKAZE
    kp = akaze.detect(img, None)

    # compute the descriptors with AKAZE
    kp, des = akaze.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(
        img,
        kp,
        None,
        color=(0, 255, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv.imshow("AKAZE", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
