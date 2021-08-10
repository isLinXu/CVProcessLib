# https://nanonets.com/blog/optical-flow/

import sys
import os

import cv2
import numpy as np

# if len(sys.argv) <= 1:
#     print(f"Format to call: {sys.argv[0]} video_stream_file/camera_index")
#     exit(os.EX_IOERR)
#
# video_stream = sys.argv[1]

# print(f"Opening video stream {video_stream}")
video_stream = 0
cap = cv2.VideoCapture(video_stream)

if not cap.isOpened():
    print(f"Error opening video stream: {video_stream}")

    camera_index = int(sys.argv[1])
    print(f"Opening camera index {camera_index}")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error opening camera index {camera_index}")
        exit(os.EX_IOERR)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame width: {frame_width}")
print(f"Frame height: {frame_height}")

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Dense optical flow", cv2.WINDOW_NORMAL)

# Parameters for Farnerback optical flow
feature_params = dict(
    pyrScale=0.5, numLevels=3, winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0
)

# Create Farnerback optical flow
optical_flow = cv2.FarnebackOpticalFlow_create(**feature_params)

# First frame in the sequence
ret, first_frame = cap.read()

# Converts frame to gray
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Creates an image filled with zeroes for drawing purposes
mask = np.zeros_like(first_frame)

# Sets mask saturation to maximum
mask[..., 1] = 255

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converts each frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculates dense optical flow by Farnerback method
    flow = optical_flow.calc(prev_gray, gray, None)

    # Compute the magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets the image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # Displays output frame
    cv2.imshow("Dense optical flow", rgb)

    # Update previous frame
    prev_gray = gray.copy()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()

print("Done.")

cv2.destroyAllWindows()
