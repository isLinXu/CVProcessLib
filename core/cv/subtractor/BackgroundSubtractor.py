# coding:utf-8

import cv2

# 获取摄像头
camera = cv2.VideoCapture(0)
# 获取背景分割器对象
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)

while True:
    # 读取帧
    ret, frame = camera.read()
    # 获取前景
    fgmask = bs.apply(frame)
    # 对前景二值化
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    # 膨胀运算
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    # 检测轮廓
    contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 将轮廓画在原图像上
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (2555, 255, 0), 2)
    # 显示前景
    cv2.imshow("fgmask", fgmask)
    # 显示二值化
    cv2.imshow("thresh", th)
    # 显示带有轮廓的原图
    cv2.imshow("detection", frame)
    if cv2.waitKey(5) & 0xff == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()