import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize

## Global =================================

VIDEO_PATH = '/home/linxu/Desktop/视频防抖/video.mp4'
# The larger the more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS = 50
ZOOM_FACTOR = 1.3


## Function ===============================

def errSq(X, pts_old, pts_new):
    # Squarred sum of errors to minimize
    # Function is to be called by scipy.minimize
    T = X

    sq_sum = 0
    for pt1, pt2 in zip(pts_old, pts_new):
        sq_sum += (np.sum(np.abs(pt1[0] + T - pt2[0]))) ** 2

    return sq_sum


def estimateTranslation(pts_old, pts_new):
    # Setup for scipy.minimize
    arguments = (pts_old, pts_new,)

    cons = ({'type': 'ineq',
             'fun': errSq,
             'args': arguments
             })

    # Find optimal translation
    result = minimize(errSq, [[0, 0]], method='SLSQP', args=arguments, constraints=cons)
    return result.x


def slightZoom(frame, zoom_factor):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, zoom_factor)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def avgMovement(curve, radius):
    window_size = 2 * radius + 1
    # New filter
    f = np.ones(window_size) / window_size
    # Add padding
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]

    return curve_smoothed


def smoothTraj(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(2):
        smoothed_trajectory[:, i] = avgMovement(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def plotTrajectory(traj):
    X_pos = [pos[0] for pos in traj]
    Y_pos = [pos[1] for pos in traj]

    plt.plot(X_pos, Y_pos)
    plt.show()


## Main ===============================

def main():
    # Open video and get information
    global frame_stabilized
    cap = cv2.VideoCapture(VIDEO_PATH)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video codec
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (2 * w, h))
    out2 = cv2.VideoWriter('video_stable.mp4', fourcc, fps, (w, h))

    # Save first frame
    ret, prev = cap.read()

    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 2), np.float32)

    for i in range(n_frames - 2):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # Read next frame
        success, curr = cap.read()
        if not success:
            break

            # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Sanity check
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        # pts struct : [[[ptx, pty]], [[ptx, pty]]]

        # Find transformation matrix
        T = estimateTranslation(prev_pts, curr_pts)

        # Store transformation
        transforms[i] = T

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smoothTraj(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = 1
        m[0, 1] = 0
        m[1, 0] = 0
        m[1, 1] = 1
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        frame_stabilized = slightZoom(frame_stabilized, ZOOM_FACTOR)

        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        if (frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2));

        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)
        out.write(frame_out)

        out2.write(frame_stabilized)
    plotTrajectory(trajectory)

    # Release video
    cap.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
