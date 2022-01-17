import cgi
import json
from json import *

form = cgi.FieldStorage()

'''通过form.value可以直接获取post的json字符串，
继续使用json.loads（）函数将json字符串转换成python unicode对象
JSONDecoder().decode（）可以将json字符串转换成python 的dict类型
'''
json_str = json.loads(form.value)

json_dict = JSONDecoder().decode(json_str)