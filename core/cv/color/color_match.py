
import cv2
import numpy as np


# 根据颜色提取对应色域
def get_find_Color(img, lower = (0,0,0), higher= (255,255,255)):
    lowHue = lower[0]
    lowSat = lower[1]
    lowVal = lower[2]
    highHue = higher[0]
    highSat = higher[1]
    highVal = higher[2]

    # 显示原始图像。
    cv2.imshow('frame', img)

    # 可选择不同的模糊方法
    frameBGR = cv2.GaussianBlur(img, (7, 7), 0)
    # frameBGR = cv2.medianBlur(frameBGR, 7)
    # frameBGR = cv2.bilateralFilter(frameBGR, 15 ,75, 75)
    # kernal = np.ones((15, 15), np.float32)/255
    # frameBGR = cv2.filter2D(frameBGR, -1, kernal)
    # cv2.imshow('frameBGR_kernal', frameBGR)

    # 显示模糊图像。
    cv2.imshow('blurred', frameBGR)
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

    # 定义HSV值颜色范围
    colorLow = np.array([lowHue, lowSat, lowVal])
    colorHigh = np.array([highHue, highSat, highVal])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    # 显示mask
    cv2.imshow('mask-plain', mask)

    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    # 显示形态变换遮罩
    cv2.imshow('mask', mask)

    # 将遮罩放在原始图像的上方。
    result = cv2.bitwise_and(img, img, mask=mask)

    # 显示最终输出图像
    cv2.imshow('colorTest', result)
    cv2.waitKey()
    return mask

# 根据颜色提取对应色域
def getColorMask(img, lower = (0,0,0), higher= (255,255,255)):
    '''
    根据颜色提取对应色域
    :param img:
    :return:
    # colorLow [0 0 1]
    # colorHigh [  0 255 255]
    '''
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 掩模提取色域
    mask = cv2.inRange(img_hsv, lower, higher)
    # mask = cv2.medianBlur(mask, 3)
    # cv2.imshow('mask-plain', mask)

    # kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

    # 显示形态变换遮罩
    # cv2.imshow('mask', mask)

    return mask

if __name__ == '__main__':
    # img_path = '/home/linxu/Desktop/南宁电厂项目/OCR/0_0_0_15_0_0/1000001_20220812230113_v.jpeg'
    img_path = '/home/linxu/Desktop/山西焦化项目/test_image/2.png'
    src = cv2.imread(img_path)
    cv2.imshow('src', src)

    # lower = (0, 70, 152)
    # higher = (255, 255, 255)
    lower = (0, 74, 200)
    higher = (255, 255, 255)
    # colorLow [  0  74 200]
    # colorHigh [255 255 255]
    # lower = (0, 70, 140)
    # higher = (255, 140, 255)

    # lower = (0,40,117)
    # higher = (255,196,255)

    # lower = (0, 76, 135)
    # higher = (255, 195, 255)

    # lower = (0, 0 ,1)
    # higher = (0,255,255)


    mask_red = getColorMask(src, lower, higher)
    cv2.imshow('mask_red', mask_red)
    # cv2.waitKey()

    # import cv2
    # import numpy as np
    #
    # gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #
    contours, hierarchy = cv2.findContours(mask_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(src,contours,-1,(0,0,255),3)

    cv2.imshow("img", src)
    cv2.waitKey(0)



    # img_file = r"D:\data\timg.jpg"
    # img = cv2.imread(img_file)
    points = contours
    # print('contours', contours)
    # points = np.array([[154, 154],[253, 171], [154, 176],[248, 204]], np.int32)  # 数据类型为int32或者float32
    # print(points)
    # points = points.reshape(4, 1, 2)  #注意shape必须为n*1*2
    # print(points)
    # area, triangle = cv2.minEnclosingTriangle(points)
    # print(area, triangle, triangle.shape)
    # for i in range(3):
    #     point1 = triangle[i, 0, :]
    #     point2 = triangle[(i+1)%3, 0, :]
    #     # print(point1)
    #     cv2.line(src, tuple(point1), tuple(point2), (0, 0, 255), 2)
    #
    # cv2.imshow("img", src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area, trg1 = cv2.minEnclosingTriangle(contours[0])
    # print(area)
    print(trg1)
    for i in range(0, 3):
        cv2.line(src, tuple(trg1[i][0]), tuple(trg1[(i + 1) % 3][0]), (0, 255, 0), 2)

        cv2.imshow("img2", src)
        cv2.waitKey()
    # cv2.destroyAllWindows()
