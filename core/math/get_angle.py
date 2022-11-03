import cv2
import math

path = 'test.jpg'
img = cv2.imread(path)
pointsList = [] #定义点参数列表

# 定义鼠标点函数
def mousePoints(event,x,y,flags,params):
    # 如果事件是鼠标左键按下，将会记录x和y
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        # 将一二两点，和一三两点进行连线
        if size != 0 and size % 3 != 0:
            cv2.line(img,tuple(pointsList[round((size-1)/3)*3]),(x,y),(0,0,255),2)
        cv2.circle(img,(x,y),5,(0,0,255),cv2.FILLED) # 在鼠标点出对应画上一个实心圆
        pointsList.append([x,y]) # 进行点的附加，每点击一次，点的坐标进行累加
        # print(pointsList)
        # print(x,y)

# 定义梯度函数
def gradient(pt1,pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])

def getAngle(pointslist):
    pt1, pt2, pt3 = pointslist[-3:] # 将点列表的各点对应赋值给变量pt1,pt2,pt3
    m1 = gradient(pt1,pt2) # 第一点和第二点之间的斜率
    m2 = gradient(pt1,pt3) # 第一点和第三点之间的斜率
    angR = math.atan((m2-m1)/(1+(m2*m1))) # 角度对应的tan值
    angD = round(math.degrees(angR)) # 反解出对应角度值
    # 在img图像中打印文本—-对应角度的绝对值
    cv2.putText(img,str(abs(angD)),(pt1[0]-40,pt1[1]-20),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),2)
    # print(angD)

# 进行while循环
while True:
    # 点列表长度余3为0，调用函数getAngle()
    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)


    cv2.imshow('Image',img)
    # 鼠标回调值，返回对应鼠标坐标值
    cv2.setMouseCallback('Image',mousePoints)
    # 此时waitKey()参数改为1，图像才会进行更新
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList = [] # 刷新我们的点列表
        img = cv2.imread(path) # 图像变为原始图像
