# -*- coding: utf-8 -*-
#
# 文字相关
# Author: alex
# Created Time: 2020年05月23日 星期六 17时05分27秒
import os
from PIL import ImageFont, ImageDraw

font = None
package_path = os.path.dirname(os.path.realpath(__file__))
default_font_path = os.path.join(package_path, 'SimHei.ttf')


def get_default_font_path():
    """获取默认的字体文件路径
    :return default_font_path str
    """
    return default_font_path


def set_font(font_size=12, font_path=None, encoding='utf-8'):
    """设置字体
    :param font_path str|None 字体路径，默认则使用的字体是：SimHei
    :param font_size int 字体大小，默认为12
    :param encoding str 字体编码，默认为utf-8
    :return font ImageFont.truetype
    """
    global font
    if font_path is None:
        font_path = default_font_path

    font = ImageFont.truetype(font_path, font_size, encoding=encoding)
    return font


def add_chinese(draw, pos, text, fill=(255, 0, 0)):
    """往图像上添加中文
    注意：在执行该函数之前，需要先初始化字体，对应函数：set_font
    :param draw ImageDraw.Draw(img)
    :param pos list|tuple 显示文字的位置，格式: (x, y)
    :param text str 需要显示的文字
    """
    global font
    if font is None:
        font = set_font()

    draw.text(pos, text, font=font, fill=fill)
    return draw


def add_chinese_img(img, pos, text, fill=(255, 0, 0)):
    """往图像上添加中文
    注意：在执行该函数之前，需要先初始化字体，对应函数：set_font
    :param img PIL图像格式
    :param pos list|tuple 显示文字的位置，格式: (x, y)
    :param text str 需要显示的文字
    """
    draw = ImageDraw.Draw(img)
    return add_chinese(draw, pos, text, fill=fill)
