import cv2
from vidstab import VidStab, layer_blend
import numpy as np
import time

stabilizer = VidStab()
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture('/home/linxu/Desktop/视频防抖/video.mp4')
kernel = np.ones((3, 3), np.uint8)
t1 = time.time()
# cap.set(3, 1280)  # width=1920
# cap.set(4, 720)  # height=1080

while True:
    x1_, y1_, x2_, y2_ = [], [], [], []
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,
                                                  layer_func=layer_blend,
                                                  border_size=100,
                                                  border_type='black',
                                                  smoothing_window=20)
    if stabilized_frame is None:
        break
    t2 = time.time()
    if t2 - t1 > 3:
        stabilized_copy = stabilized_frame.copy()
        # stabilized_copy = cv2.resize(stabilized_copy, (640, 320) interpolation=cv2.INTER_AREA)
        stabilized_copy = cv2.cvtColor(stabilized_copy, cv2.COLOR_BGR2GRAY)
        # _, stabilized_copy = cv2.threshold(stabilized_copy, 29, 255, cv2.THRESH_BINARY)
        _, stabilized_copy = cv2.threshold(stabilized_copy, 128, 255, cv2.THRESH_BINARY)
        stabilized_copy = cv2.morphologyEx(stabilized_copy, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(stabilized_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(stabilized_copy)
                x1_.append(x)
                y1_.append(y)
                x2_.append(x + w)
                y2_.append(y + h)
            x1 = min(x1_)
            y1 = min(y1_)
            x2 = max(x2_)
            y2 = max(y2_)
        # cv2.rectangle(stabilized_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # stabilized_frame = stabilized_frame[y1:y2 - 40, x1 + 40:x2 - 40, :]
    cv2.imshow('Stabilized Frame', stabilized_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
