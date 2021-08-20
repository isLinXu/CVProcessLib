import numpy as np
import time
import cv2
import cv2.aruco as aruco

file_path = '/home/hxzh02/桌面/arucoImage_640*480/6.jpg'
# file_path = '/home/hxzh02/桌面/testIMg/1.jpg'

# 读取图片
frame = cv2.imread(file_path)
# 调整图片大小
frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_CUBIC)
# frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
# width = 640
# height = 480
# width = 600
# height = 337
# dim = (width, height)

# frame = cv2.resize(frame, None, 640, 480, interpolation=cv2.INTER_CUBIC)
# frame = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)
# 灰度话
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 设置预定义的字典
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

# 使用默认值初始化检测器参数
parameters = aruco.DetectorParameters_create()
# 使用aruco.detectMarkers()函数可以检测到marker，返回ID和标志板的4个角点坐标
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
print(ids)
# 画出标志位置
aruco.drawDetectedMarkers(frame, corners, ids)

cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
