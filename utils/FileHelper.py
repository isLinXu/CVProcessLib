# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: 912588967@qq.com
@time: 2019-09-31 上午11:30
@desc: 图片辅助操作（非处理）
'''

import os, base64
from temptest.initialize import RUN_PATH, ENV_PATH
from utils import PubHelper

def Base64SteamToFile(stream, stripstr=""):
    if stream.startswith(stripstr):
        stream = str(stream)[len(stripstr):]
    filestream = base64.b64decode(stream)
    return filestream

def SaveFile(filestream, functionname, ext, DATE_PATH):
    # 存储文件
    PATH = ENV_PATH + functionname + '/' + DATE_PATH + '/'
    SAVE_PATH = RUN_PATH + '/upload' + PATH
    isExists = os.path.exists(SAVE_PATH)
    if not isExists:
        os.makedirs(SAVE_PATH)
    # 使用通用唯一识别码命名
    filename = PubHelper.getuuid() + '.' + ext
    # 对应路径保存文件
    with open(SAVE_PATH + filename, "wb") as f:
        f.write(filestream)
    return filename, PATH