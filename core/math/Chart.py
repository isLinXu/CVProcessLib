# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: whang@ikeeko.com
@time: 2019-05-28 上午11:23
@desc: 图表技术
    表：
        chartlist的结构为[[x11, x12, x13, ...], [x21, x22, x23, ...], ...]

'''

# ----------------------表检索函数-------------------------
import math
from utils.ListHelper import listPerm
from utils.FuncHelper import timereport

@timereport
def chartValueRoute(chartlist, axis=0, taketimeinfo=[], fname='二维表匹配对应'):
    """在二维表格中查找不同行不同列的最小/大路径位置（可用做数据中两两之间最小/大距离路径计算)"""
    """
        输入：chartlist = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        输出：total = 34
             pos = [0, 1, 2, 3]
             value = [1, 6, 11, 16]
    """
    # 初始化反馈
    total = math.inf
    pos = []
    value = []
    # 行位个数
    row = len(chartlist)

    # 按行全排列输出
    x = [i for i in range(row)]
    x = listPerm(x)

    # 按全排列遍历所有可能出现情况，找出一对最小路径值
    for i in x:
        totallist = []
        for j in range(row):
            totallist.append(chartlist[j][i[j]])
        ttotal = sum(totallist)
        if ttotal < total:
            total = ttotal
            pos = i
            value = totallist
    return total, pos, value

# if __name__ == '__main__':
#     chartlist = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
#     print(ChartValueRoute(chartlist, taketimeinfo=[]))