'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-07-11 上午11:54
@desc: CAMShift运动跟踪算法

在视频或者是摄像头当中，如果被追踪的物体迎面过来，由于透视效果，物体会放大。之前设置好的窗口区域大小会不合适。
OpenCV实验室实现了一个CAMshift算法，首先使用meanshift算法找到目标，然后调整窗口大小，
而且还会计算目标对象的的最佳外接圆的角度，并调整窗口。并使用调整后的窗口对物体继续追踪。
使用方法与meanShift算法一样，不过返回的是一个带有旋转角度的矩形。
Camshift，连续的自适应MeanShift算法，是对MeanShift算法的改进算法，
可以在跟踪的过程中随着目标大小的变化实时调整搜索窗口大小，
对于视频序列中的每一帧还是采用MeanShift来寻找最优迭代结果，至于如何实现自动调整窗口大小的，可以查到的论述较少，
我的理解是通过对MeanShift算法中零阶矩的判断实现的。
'''

import cv2
import numpy as np

xs, ys, ws, hs = 0, 0, 0, 0  # selection.x selection.y
xo, yo = 0, 0  # origin.x origin.y
selectObject = False
trackObject = 0


def onMouse(event, x, y, flags, prams):
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject
    if selectObject == True:
        xs = min(x, xo)
        ys = min(y, yo)
        ws = abs(x - xo)
        hs = abs(y - yo)
    if event == cv2.EVENT_LBUTTONDOWN:
        xo, yo = x, y
        xs, ys, ws, hs = x, y, 0, 0
        selectObject = True
    elif event == cv2.EVENT_LBUTTONUP:
        selectObject = False
        trackObject = -1


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.namedWindow('imshow')
    cv2.setMouseCallback('imshow', onMouse)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while (True):
        ret, frame = cap.read()
        if trackObject != 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))
            if trackObject == -1:
                track_window = (xs, ys, ws, hs)
                maskroi = mask[ys:ys + hs, xs:xs + ws]
                hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
                roi_hist = cv2.calcHist([hsv_roi], [0], maskroi, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                trackObject = 1
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            dst &= mask
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)

        if selectObject == True and ws > 0 and hs > 0:
            cv2.imshow('imshow1', frame[ys:ys + hs, xs:xs + ws])
            cv2.bitwise_not(frame[ys:ys + hs, xs:xs + ws], frame[ys:ys + hs, xs:xs + ws])
        cv2.imshow('imshow', frame)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()
