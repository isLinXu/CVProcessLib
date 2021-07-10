# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: 912588967@qq.com
@time: 2019-09-31 上午11:30
@desc: 图片辅助操作（非处理）
'''

import os, cv2, base64
import numpy as np
from temp.initialize import RUN_PATH
from utils import PubHelper

def cv2ImgToBytes(img):
    # 如果直接tobytes写入文件会导致无法打开，需要编码成一种图片文件格式(jpg或png)，再tobytes
    # 这里得到的bytes 和 with open("","rb") as f: bytes=f.read()的bytes可能不一样，
    # 如果用这里得到的bytes保存过一次，下次就f.read()和cv2ImgToBytes(img)会一样
    return cv2.imencode('.png', img)[1].tobytes()

# def bytesToCv2Img(bytes):
#     return cv2.imdecode(np.fromstring(r.content,"uint8"), 1)

def Y_channel_to_gray(stream):
    # 图像解码
    stream = str(stream).replace('data:image/jpg;base64,', '').replace('data:image/png;base64,', ''). \
        replace('data:image/jpeg;base64,', '')
    stream = base64.b64decode(stream)
    img_array = np.fromstring(stream, dtype=np.uint8)
    img_array_reshape = np.reshape(img_array, (480, 640))
    return img_array_reshape

def Base64SteamToImg(stream):
    # 图像解码
    stream = str(stream).replace('data:image/jpg;base64,', '').replace('data:image/png;base64,', ''). \
        replace('data:image/jpeg;base64,', '')
    stream = base64.b64decode(stream)
    img = np.fromstring(stream, dtype=np.uint8)
    # 判断是否为单通道图，如果为单通道转为BGR模式，兼容BGR和Y模式
    if img.shape[0] == 480*640:
        img = np.reshape(img, (480, 640))
        img = cv2.flip(img, 1)
        return img
    img = cv2.imdecode(img, -1)
    return img

def Base64SteamToByteSteam(stream):
    stream = str(stream).replace('data:image/jpg;base64,', '').replace('data:image/png;base64,', ''). \
        replace('data:image/jpeg;base64,', '')
    bytestream = bytes(stream, encoding="utf-8")
    return bytestream

def SaveImg(img, functionname, DATE_PATH, imgformat='png'):
    # 存储图片
    CV_PATH_img = ENV_PATH + functionname + '/' + DATE_PATH + '/img/'
    CV_PATH_thumbimg = ENV_PATH + functionname + '/' + DATE_PATH + '/thumbimg/'
    SAVE_PATH_img = RUN_PATH + '/upload' + CV_PATH_img
    isExists_img = os.path.exists(SAVE_PATH_img)
    SAVE_PATH_thumbimg = RUN_PATH + '/upload' + CV_PATH_thumbimg
    isExists_thumbimg = os.path.exists(SAVE_PATH_thumbimg)
    if not isExists_img:
        os.makedirs(SAVE_PATH_img)
    if not isExists_thumbimg:
        os.makedirs(SAVE_PATH_thumbimg)
    # 使用通用唯一识别码命名
    filename = PubHelper.getuuid() + '.' + str(imgformat)
    # 对应路径保存文件
    cv2.imwrite(SAVE_PATH_img + filename, img)
    qrthumbimg = cv2.resize(img, (128, 128))
    cv2.imwrite(SAVE_PATH_thumbimg + filename, qrthumbimg)
    saveimginfo = {'filename': filename, 'filepath': CV_PATH_img,
                   'thumbfilepath': CV_PATH_thumbimg}
    return saveimginfo

def SaveTempImg(img, functionname, imgformat='png'):
    # 存储图片
    CV_PATH_img = functionname + '/img/'
    CV_PATH_thumbimg = functionname + '/thumbimg/'
    SAVE_PATH_img = RUN_PATH + '/upload' + CV_PATH_img
    isExists_img = os.path.exists(SAVE_PATH_img)
    SAVE_PATH_thumbimg = RUN_PATH + '/upload' + CV_PATH_thumbimg
    isExists_thumbimg = os.path.exists(SAVE_PATH_thumbimg)
    if not isExists_img:
        os.makedirs(SAVE_PATH_img)
    if not isExists_thumbimg:
        os.makedirs(SAVE_PATH_thumbimg)
    # 使用通用唯一识别码命名
    filename = PubHelper.getuuid() + '.' + str(imgformat)
    # 对应路径保存文件
    cv2.imwrite(SAVE_PATH_img + filename, img)
    qrthumbimg = cv2.resize(img, (128, 128))
    cv2.imwrite(SAVE_PATH_thumbimg + filename, qrthumbimg)
    savetempimginfo = {'filename': filename, 'filepath': CV_PATH_img, 'thumbfilepath': CV_PATH_thumbimg}
    return savetempimginfo

def SaveTrainImg(img, functionname, categoryname):
    # 存储图片
    CV_PATH_img = functionname + '/' + categoryname + '/'
    SAVE_PATH_img = RUN_PATH + '/upload' + CV_PATH_img
    isExists_img = os.path.exists(SAVE_PATH_img)
    if not isExists_img:
        os.makedirs(SAVE_PATH_img)
    # 使用通用唯一识别码命名
    filename = PubHelper.getuuid() + '.png'
    # 对应路径保存文件
    cv2.imwrite(SAVE_PATH_img + filename, img)
    savetrainimginfo = {'filename': filename, 'filepath': CV_PATH_img}
    return savetrainimginfo