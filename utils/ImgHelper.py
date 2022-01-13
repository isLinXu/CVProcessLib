import os

import cv2
from PIL import Image
import numpy


def PIL_image_to_Opencv_image(path):
    '''
    PIL.Image格式转换为OpenCV BGR格式
    :param path:
    :return:
    '''
    image = Image.open(path)
    image.show()
    img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("OpenCV", img)
    cv2.waitKey()
    return img

def Opencv_image_to_PIL_image(path):
    '''
    OpenCV BGR格式转换为PIL.Image格式
    :param path:
    :return:
    '''
    img = cv2.imread(path)
    cv2.imshow("OpenCV",img)
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    image.show()
    cv2.waitKey()
    return image


def png2jpg(path):
    print(path)
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1] == '.png':
            # print(filename)
            img = cv2.imread(path + filename)
            print(filename.replace(".png", ".jpg"))
            newfilename = filename.replace(".png", ".jpg")
            cv2.imshow("Image",img)
            cv2.waitKey(0)
            cv2.imwrite(path + newfilename, img)

if __name__ == '__main__':
    path = '/media/hxzh02/SB@home/hxzh/Dataset/Plane_detect_datasets/VOCdevkit_lineextract_detect/VOC2007/JPEGImages/'
    png2jpg(path)