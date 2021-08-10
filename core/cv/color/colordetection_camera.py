import cv2
import numpy as np

video = cv2.VideoCapture(0)

#设置相机参数
frameWidth = 640
frameHeight = 480
brightness = 150

video.set(3, frameWidth)
video.set(4, frameHeight)
video.set(10, brightness)

def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)

# 初始化滚动条变量
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 255, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

# 获取一种特定颜色的遮罩
cv2.createTrackbar("Hue Min", "TrackBars", 44, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 51, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 159, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 186, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 97, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    success, img = video.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask = mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])

    cv2.imshow("Horizontal stacking", hStack)
    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()