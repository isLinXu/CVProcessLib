# https://nanonets.com/blog/optical-flow/

import sys
import os

import cv2
import numpy as np

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=2, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 0.03),
)

# Variable for color to draw optical flow track
color = (0, 255, 0)

# if len(sys.argv) <= 1:
#     print(f"Format to call: {sys.argv[0]} video_stream_file/camera_index")
#     exit(os.EX_IOERR)
#
# video_stream = sys.argv[1]

video_stream = 0
print(f"Opening video stream {video_stream}")

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

# First frame in the sequence
ret, first_frame = cap.read()

# Converts frame to gray
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Find the strongest corners with Shi-Tomasi method
prev_features = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Creates an image filled with zeroes for drawing purposes
mask = np.zeros_like(first_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converts each frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculates sparse optical flow by Lucas-Kanade method
    next_features, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_features, None, **lk_params
    )

    # Select good feature points for previous position
    good_old = prev_features[status == 1]

    # Select good feature points for next position
    good_new = next_features[status == 1]

    # Draws optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a continuous flattened array as (x,y) coordinates for new point
        a, b = new.ravel()
        # Returns a continuous flattened array as (x,y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position
        # mask = cv2.line(mask, (a, b), (c, d), color, 2)
        # Draws filled circle at new position
        frame = cv2.circle(frame, (a, b), 3, color, -1)

    # Overlays the optical flow tracks on the original frame
    output = cv2.add(frame, mask)

    # Update previous good feature points
    prev_features = good_new.reshape(-1, 1, 2)

    # Update previous frame
    prev_gray = gray.copy()

    cv2.imshow("Sparse Optical Flow", output)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()

print("Done.")

cv2.destroyAllWindows()
