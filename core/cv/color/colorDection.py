'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-05-26 上午11:54
@desc: 鼠标选择色块选择打印颜色数值
'''

import cv2
def mouse_click(event, x, y, flags, para):
    global img
    global hsv
    global gray
    global pointIndex
    global pts
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        print('PIX:', x, y)
        print("BGR:", img[y, x])
        print("GRAY:", gray[y, x])
        print("HSV:", hsv[y, x])
        print("------------------------")

def show_window():
    while True:
        cv2.imshow('img', img)
        if (cv2.waitKey(20) & 0xFF == 27):
            break

if __name__ == '__main__':
    img = cv2.imread('/home/linxu/PycharmProjects/CVProcessLib/images/呼吸器/4.jpeg')
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", mouse_click)
    show_window()
    cv2.destroyAllWindows()