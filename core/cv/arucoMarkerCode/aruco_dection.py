import numpy as np
import time
import cv2
import cv2.aruco as aruco
# 1 8 9 10 11 12 13 14 15 16 20 21 22
file_path = '/home/linxu/Desktop/CaptureFiles/2021-09-06/1.jpg'

# 读取图片
frame = cv2.imread(file_path)
# 调整图片大小
frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

# 灰度化
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 设置预定义的字典
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

# 使用默认值初始化检测器参数
parameters = aruco.DetectorParameters_create()
# 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
print('ids', ids)
print('corners', corners)

if ids != None and len(corners) != 0:
    # 画出标志位置
    aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("frame", frame)
    cv2.waitKey(0)

    print('corners[0]', corners[0][0][0])
    corners = np.uint16(np.around(corners))
    point1 = (corners[0][0][0][0], corners[0][0][0][1])
    point2 = (corners[0][0][1][0], corners[0][0][1][1])
    point3 = (corners[0][0][2][0], corners[0][0][2][1])
    point4 = (corners[0][0][3][0], corners[0][0][3][1])
    # print('point1', point1)5

    # BGR
    # ((x1,y1), (x2,y2), (x3,y3), (x4, y4))
    cv2.circle(frame, point1, 5, (0, 0, 255), -1)
    cv2.circle(frame, point2, 5, (0, 255, 255), -1)
    cv2.circle(frame, point3, 5, (255, 0, 0), -1)
    cv2.circle(frame, point4, 5, (0, 225, 0), -1)

    # pt1 = (point1[0] + 65, point1[1] - 50)
    # pt2 = (point2[0] + 65, point1[1])
    # pt3 = (point3[0] + 50, point3[1] - 50)
    # pt4 = (point4[0] + 50, point4[1] - 50)
    # cv2.circle(frame, pt1, 5, (0, 255, 0), -1)
    # cv2.circle(frame, pt2, 5, (0, 255, 0), -1)
    # cv2.circle(frame, pt3, 5, (0, 255, 0), -1)
    # cv2.circle(frame, pt4, 5, (0, 255, 0), -1)
    # cv2.circle(frame, point4, 5, (0, 0, 255), -1)

    cv2.imshow("circle", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
