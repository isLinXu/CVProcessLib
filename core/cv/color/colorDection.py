'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-05-26 上午11:54
@desc: 鼠标选择色块选择打印颜色数值
'''

import cv2

img = cv2.imread('/home/linxu/PycharmProjects/CVProcess/images/表计/test1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def mouse_click(event, x, y, flags, para):
    if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        print('PIX:', x, y)
        print("BGR:", img[y, x])
        print("GRAY:", gray[y, x])
        print("HSV:", hsv[y, x])


if __name__ == '__main__':
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_click)

    while True:
        cv2.imshow('img', img)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()