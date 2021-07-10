# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: whang@ikeeko.com
@time: 2019-03-28 下午17:45
@desc: 线程辅助类
'''

import threading, time

# MyThread.py线程类
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        time.sleep(2)
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None