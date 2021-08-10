import cv2 as cv

img = cv.imread("Lenna.png")

cv.namedWindow("KAZE", cv.WINDOW_NORMAL)


def f(x):
    return


diffusivity_types = {
    0: cv.KAZE_DIFF_PM_G1,
    1: cv.KAZE_DIFF_PM_G2,
    2: cv.KAZE_DIFF_WEICKERT,
    3: cv.KAZE_DIFF_CHARBONNIER,
}

# Initiate KAZE detector
cv.createTrackbar("Extended", "KAZE", 0, 1, f)
cv.createTrackbar("Upright", "KAZE", 0, 1, f)
cv.createTrackbar("Diffusivity Type", "KAZE", 0, len(diffusivity_types) - 1, f)
cv.createTrackbar("Threshold", "KAZE", 10, 100, f)
cv.createTrackbar("Octaves", "KAZE", 4, 16, f)
cv.createTrackbar("Octave Layers", "KAZE", 4, 16, f)

while True:
    current_extended = cv.getTrackbarPos("Extended", "KAZE")
    current_upright = cv.getTrackbarPos("Upright", "KAZE")
    current_diffusivity_type = cv.getTrackbarPos("Diffusivity Type", "KAZE")
    current_threshold = cv.getTrackbarPos("Threshold", "KAZE") / 10000
    current_octaves = cv.getTrackbarPos("Octaves", "KAZE") + 1
    current_octave_layers = cv.getTrackbarPos("Octave Layers", "KAZE") + 1

    kaze = cv.KAZE_create(
        extended=current_extended,  # Set to enable extraction of extended (128-byte) descriptor.
        upright=current_upright,
        threshold=current_threshold,
        nOctaves=current_octaves,
        nOctaveLayers=current_octave_layers,
        diffusivity=diffusivity_types[current_diffusivity_type],
    )

    # find the keypoints with KAZE
    kp = kaze.detect(img, None)

    # compute the descriptors with KAZE
    kp, des = kaze.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(
        img,
        kp,
        None,
        color=(0, 255, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv.imshow("KAZE", img2)

    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()
