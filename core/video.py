import numpy as np
import cv2


if __name__ == '__main__':


    # Read input video
    cap = cv2.VideoCapture('/home/linxu/Desktop/视频防抖/video.mp4')

    # Get frame count        获得帧数？
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 1103帧

    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 848
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 476

    # Define the codec for output video       codec是视频编码吗？
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # MJPG格式输出的是将视频图像采用JPEG格式压缩后得到的视频帧，优点是帧率高（视频开启快，曝光快），缺点是影像有马赛克，并且需要解码器，会占用PC系统资源。MJPG视频帧直接保存成jpg文件即可用常见的图片查看工具打开。
    # MJPG是一种视频输出格式

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率？

    # Set up output video
    out = cv2.VideoWriter('compete_video.mp4', fourcc, fps, (2 * w, h))  # 到底乘不乘以2
    out2 = cv2.VideoWriter('stabilized_video.mp4', fourcc, fps, (w, h))  # 到底乘不乘以2

    # 读取第一帧prev
    _, frame = cap.read()

    # 第一帧转化成灰度图，类型是ndarray
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # prev_gray是ndarray

    # 对于视频稳定，我们需要捕捉视频的两帧，估计帧之间的运动，最后校正运动。
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames - 1, 3), np.float32)  # n_frames是总帧数。n_frames - 1行，3列的0

    # 在3.1步骤中，我们在前一帧中找到了一些好的特征。在步骤3.2中，我们使用光流来跟踪特征。 换句话说，我们已经找到了特征在当前帧、前一帧中的位置
    # 可以使用这两组点来找到映射前一个坐标系到当前坐标系的刚性（欧几里德）变换，这是使用函数estimateRigidTransform完成的。
    # 一旦我们估计了运动，我们可以把它分解成x和y的平移和旋转（角度）。我们将这些值存储在一个数组中，这样就可以【平稳地】更改它们。

    for i in range(n_frames - 2):
        # 找到前一帧的特征点
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # 读取下一帧
        success, curr = cap.read()
        if not success:
            break

        # 转化成灰度图
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (i.e. track feature points)  使用Lucas-Kanade光流算法在下一帧（当前帧）中跟踪特征点pts
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        # status ：输出状态向量(无符号字符); 如果找到相应特征的流，则向量的每个元素设置为1，否则设置为0

        # Sanity check  完整性检查，通常用在对输入数据的检查。匹配特征点大小
        assert prev_pts.shape == curr_pts.shape

        # Filter only valid points  获得有效点？
        idx = np.where(status == 1)[0]  # ???
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix  使用这两组点来找到映射前一个坐标系到当前坐标系的刚性（欧几里德）变换
        m, inl = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        # estimateRigidTransform这个函数被弃用了，只能在opencv3及以下用，https://www.cnpython.com/qa/227497
        # 改用estimateAffinePartial2D：https://blog.csdn.net/qingfengxd1/article/details/109099556

        # Extract translation  摘录翻译？
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle   旋转角度
        da = np.arctan2(m[1, 0], m[0, 0])

        # Store transformation
        transforms[i] = [dx, dy, da]

        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    # 第四步：计算帧之间的平滑运动
    # 在前面的步骤中，我们估计帧之间的运动并将它们存储在一个数组中。我们现在需要通过叠加上一步估计的微分运动来找到运动轨迹。

    # 步骤4.1:轨迹计算
    # 在这一步，我们将增加运动之间的帧来计算轨迹。我们的最终目标是平滑这条轨迹。
    # Python 在Python中，可以很容易地使用numpy中的cumsum(累计和)来实现。
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)


    # 步骤4.2:计算平滑轨迹。详细去看原文
    # 在上一步中，我们计算了运动轨迹。所以我们有三条曲线来显示运动(x, y，和角度)如何随时间变化。
    # 在这一步，我们将展示如何平滑这三条曲线。
    # 定义了一个移动平均滤波器，它接受任何曲线(即1-D的数字)作为输入，并返回曲线的平滑版本。
    def movingAverage(curve, radius):
        window_size = 2 * radius + 1
        # Define the filter
        f = np.ones(window_size) / window_size
        # Add padding to the boundaries
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        # Apply convolution
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        # Remove padding
        curve_smoothed = curve_smoothed[radius:-radius]
        # return smoothed curve
        return curve_smoothed


    # 我们还定义了一个函数，它接受轨迹并对这三个部分进行平滑处理。
    SMOOTHING_RADIUS = 50  # 为什么设为50呢？设为30效果会不会好一点？


    def smooth(trajectoryy):
        smoothed_trajectory = np.copy(trajectoryy)
        # Filter the x, y and angle curves
        for ii in range(3):
            smoothed_trajectory[:, ii] = movingAverage(trajectoryy[:, ii], radius=SMOOTHING_RADIUS)
        return smoothed_trajectory


    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # 步骤4.3:计算平滑变换

    # Calculate difference in smoothed_trajectory and trajectory
    smoothed_trajectory = smooth(trajectory)  # 这我自己加的****
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # 第五步:将平滑的摄像机运动应用到帧中
    # 差不多做完了。现在我们所需要做的就是循环帧并应用我们刚刚计算的变换。

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    # 步骤5.1:修复边界伪影
    # def fixBorder(frame):
    #     s = frame.shape
    #     # Scale the image 4% without moving the center
    #     T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    #     frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    #     return frame


    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts  修复边界瑕疵
        # frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])  # 拼接函数

        # If the image is too big, resize it.     这样视频可能会被裁剪
        if frame_out.shape[1] > 1920:
            frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2))

        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)
        out.write(frame_out)
        # 把稳定后的视频单独保存一份
        out2.write(frame_stabilized)
    cap.release()
    out.release()
    out2.release()
    cv2.destroyAllWindows()
