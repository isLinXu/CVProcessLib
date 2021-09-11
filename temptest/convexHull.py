import cv2

img = cv2.imread("/home/hxzh02/图片/11.png")
cv2.imshow("img", img)

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('binary', binary)
cv2.drawContours(img,contours, -1, (0,0,255))
cv2.imshow('draw', img)
cv2.waitKey()

hull = cv2.convexHull(contours[0], returnPoints=False)

defects = cv2.convexityDefects(contours[0], hull)
print(defects)

# for i in range(defects.shape[0]):
#     s, e, f, d = defects[i, 0]
#     start = tuple(contours[0][s][0])
#     end = tuple(contours[0][e][0])
#     far = tuple(contours[0][f][0])
#     cv2.line(img, start, end, [0, 255, 0], 2)
#     cv2.circle(img, far, 5, [0, 0, 255], -1)
#
# cv2.imshow("img1", img)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
