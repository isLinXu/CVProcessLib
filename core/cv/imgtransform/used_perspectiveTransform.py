'''
@author: linxu
@contact: 17746071609@163.com
@time: 2022-06-30 17:12 PM
@desc: 根据
'''

import cv2
import numpy as np


def img_perspect_transform(image_src, select=0, AR=(640, 1280)):
    oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])
    if select == 0:
        pts = [(450, 715), (680, 730), (420, 820), (630, 830)]
    else:
        pts = [(388, 635), (621, 654), (357, 747), (566, 762)]
    ippts = np.float32(pts)
    Map = cv2.getPerspectiveTransform(ippts, oppts)
    warped_dst = cv2.warpPerspective(image_src, Map, (AR[1], AR[0]))
    return warped_dst


if __name__ == '__main__':
    path = '/home/linxu/Desktop/计量中心数据集/电表流水线0/05.png'
    # path = '/home/linxu/Desktop/计量中心数据集/电表流水线1/04.png'
    img = cv2.imread(path)

    dst = img_perspect_transform(img)
    cv2.imshow("output", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()