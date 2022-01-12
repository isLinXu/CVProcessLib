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