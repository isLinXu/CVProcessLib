# 导入必要的包
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2


# 加载图像
path = '/home/linxu/Desktop/南宁电厂项目/OCR/0_0_0_15_0_0/1000001_20220812230113_v.jpeg'
image = cv2.imread(path)

# 1. LCD边缘可见
# 预处理步骤：保持宽高比的缩放，转换灰度，高斯模糊以减少高频噪音，Canny边缘检测器计算边缘图
# image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)
cv2.imshow('edged', edged)
cv2.waitKey()

# 2. 提取LCD本身
# 在边缘图中寻找轮廓，并按面积大小倒序排列
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

# 遍历轮廓
for c in cnts:
    # 应用轮廓近似
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 如果边缘有4个顶点（vertices），则找到了恒温器并展示
    if len(approx) == 4:
        displayCnt = approx
        break

