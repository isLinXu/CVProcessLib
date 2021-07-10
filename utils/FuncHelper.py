# -*- coding: utf-8 -*-
'''
@author: tuweifeng
@contact: 907391489@qq.com
@time: 2019-04-01 下午20:07
@desc:
'''

import time, inspect, signal, functools, traceback

def timeout(seconds, error_result=""):
    def decorated(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except:
                print(traceback.format_exc())
                result = error_result
                pass
            finally:
                signal.alarm(0)
            return result
        return functools.wraps(func)(wrapper)
    return decorated

# def timereport(f):
#     def setF(*args, **kw):
#         taketimeinfo = kw['taketimeinfo']
#         fname = inspect.getargspec(f).defaults[-1]
#         start = int(round(time.time() * 1000))
#         result = f(*args, **kw)
#         end = int(round(time.time() * 1000))
#         taketimeinfo.append({'name': fname + '时间', 'time': str(end - start) + 'ms'})
#         return result, taketimeinfo
#     return setF

def timereport(f):
    """
    :param f: 使用该装饰器的函数可选参需含:
              fname='xxx':  统计的时间片命名
              taketimeinfo=[]：  储存函数经历的时间片名及其时长，append形式添加，若为（'总时长:'）则insert于开头
    :return:  以元组方式返回使用该装饰器的函数的返回参，不影响原函数
    """
    def setF(*args, **kw):
        fname = inspect.getargspec(f).defaults[-1]
        start = int(round(time.time() * 1000))
        result_tuple = f(*args, **kw)
        end = int(round(time.time() * 1000))
        if fname == '总时间':
            result_list = list(result_tuple)
            result_list[-1].insert(0, {'name': fname, 'time': str(end - start) + 'ms'})
            result = tuple(result_list)
            return result
        elif 'taketimeinfo' not in dict(**kw).keys():
            return result_tuple
        else:
            taketimeinfo = kw['taketimeinfo']
            taketimeinfo.append({'name': fname + '时间', 'time': str(end - start) + 'ms'})
            return result_tuple, taketimeinfo
    return setF
