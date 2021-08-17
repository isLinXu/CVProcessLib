'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-06-27 上午11:54
@desc: 鼠标选取ROI矩形区域并截图
'''
import cv2
import imutils

img = cv2.imread("/home/linxu/PycharmProjects/CVProcessLib/images/状态指示器/2.jpeg")
img = imutils.resize(img, width=500)

roi = cv2.selectROI(windowName="selectROI", img=img, showCrosshair=True, fromCenter=False)
x, y, w, h = roi

cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=1)
cv2.imshow("selectROI", img)

print('ROI:', roi)
ROI = img[y:y+h, x:x+w]
cv2.imshow('ROI',ROI)
cv2.waitKey(0)
cv2.destroyAllWindows()


