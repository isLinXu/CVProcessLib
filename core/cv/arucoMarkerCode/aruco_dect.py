
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-08-20 上午11:54
@desc: aruco_marker识别
'''
import numpy as np
import time
import cv2
import cv2.aruco as aruco

# mtx = np.array([
#         [2946.48,       0, 1980.53],
#         [      0, 2945.41, 1129.25],
#         [      0,       0,       1],
#         ])
# #我的手机拍棋盘的时候图片大小是 4000 x 2250
# #ip摄像头拍视频的时候设置的是 1920 x 1080，长宽比是一样的，
# #ip摄像头设置分辨率的时候注意一下
#
#
# dist = np.array( [0.226317, -1.21478, 0.00170689, -0.000334551, 1.9892] )


# 相机纠正参数

# dist=np.array(([[-0.51328742,  0.33232725 , 0.01683581 ,-0.00078608, -0.1159959]]))
#
# mtx=np.array([[464.73554153, 0.00000000e+00 ,323.989155],
#  [  0.,         476.72971528 ,210.92028],
#  [  0.,           0.,           1.        ]])
dist = np.array(([[-0.58650416, 0.59103816, -0.00443272, 0.00357844, -0.27203275]]))
newcameramtx = np.array([[189.076828, 0., 361.20126638]
                            , [0, 2.01627296e+04, 4.52759577e+02]
                            , [0, 0, 1]])
mtx = np.array([[398.12724231, 0., 304.35638757],
                [0., 345.38259888, 282.49861858],
                [0., 0., 1.]])

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

# num = 0
while True:
    ret, frame = cap.read()
    h1, w1,c1 = frame.shape
    # 读取摄像头画面
    # 纠正畸变
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    # dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w1, h1 = roi
    # dst1 = dst1[y:y + h1, x:x + w1]
    # frame = dst1
    # cv2.imshow('frame1', frame)
    # cv2.waitKey()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters_create()
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    '''
    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
    '''

    # 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    #    如果找不打id
    if ids is not None:

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        # 估计每个标记的姿态并返回值rvet和tvec ---不同
        # from camera coeficcients
        (rvec - tvec).any()  # get rid of that nasty numpy value array error

        #        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #绘制轴
        #        aruco.drawDetectedMarkers(frame, corners) #在标记周围画一个正方形

        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners)
        ###### DRAW ID #####
        cv2.putText(frame, "Id: " + str(ids), (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示结果框架
    cv2.namedWindow('frame',cv2.WINDOW_AUTOSIZE)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == 27:  # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    if key == ord(' '):  # 按空格键保存
        #        num = num + 1
        #        filename = "frames_%s.jpg" % num  # 保存一张图像
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)
