# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: whang@ikeeko.com
@time: 2018-06-04 下午18:12
@desc: list函数辅助
'''

import numpy as np

# ----------------------list相关函数-------------------------
def listPerm(alist):
    """list全排列输出"""
    """ 
        输入：alist = [1, 2, 3]
        输出：rlist = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        时间复杂度O(n^2)，有待改善
    """
    if (len(alist) <= 1):
        return [alist]
    rlist = []
    for i in range(len(alist)):
        s = alist[:i] + alist[i+1:]
        p = listPerm(s)
        for x in p:
            rlist.append(alist[i:i+1] + x)
    return rlist

def listIndexPerm(alist, index=None):
    """list按个数全排列输出"""
    """ 
        输入：alist = [1, 2, 3]
             n表示n个一组，若为None则为全部排列
        n = 1 输出：rlist = [[1], [2], [3], [3, 2]]
        n = 2 输出：rlist = [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]]
        n = None 输出：rlist = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    """
    rlist = listPerm(alist)
    if index:
        rrlist = []
        for ilist in rlist:
            if ilist[:index] not in rrlist:
                rrlist.append(ilist[:index])
        rlist = rrlist
    return rlist

def listEndPop(strlist, str):
    """list末尾连续删除str"""
    """
        输入 list = ['1','','2','','']
        输出 list = ['1','','2']
    """
    nstrlist = strlist.copy()
    length = len(nstrlist)
    for item in range(length - 1, -1, -1):
        if nstrlist[item] == str:
            nstrlist.pop()
        else:
            break
    return nstrlist

def listAllRemove(strlist, str):
    """list把含有str的元素进行删除"""
    """
        输入list = ['1','?','2','?','?']
        输出list = ['1','2']
    """
    nstrlist = strlist.copy()
    for item in nstrlist[:]:
        if item == str:
            nstrlist.remove(item)
    return nstrlist

def listFind(alist, slist):
    """list父集是否包括子集，存在则返回下标位置，不存在则返回0，Sunday算法"""
    """ 
        输入：alist = ['a', 'b', 'c', 'd'], slist = ['c', 'd']
        输出：2
    """
    alen = len(alist)
    slen = len(slist)
    pos = -1
    i = 1
    head = 0
    stop = 0
    # 先确保有物可寻
    if slist:
        # 最大下标位置 = 总个数 - 1
        while head + slen <= alen:
            while alist[head:head + i] == slist[:i]:
                if i == slen:
                    pos = head
                    stop = 1
                    break
                else:
                    i = i + 1
            # 找到标记
            if stop == 1:
                break
            # 跳下一个阶段先判断是否越界，相等取不到值，取不到下标
            if head + slen == alen:
                break
            # 跳跃查找
            jump = alist[head + slen]
            if jump in slist:
                head = head + list(reversed(slist)).index(jump) + 1
            else:
                head = head + slen + 1
    return pos

def listReplace(alist, slist, rlist):
    """alist查找子集slist，存在则被rlist替换，不存在则返回list父集,其中slist不等于rlist"""
    """ 
        输入：alist = ['a', 'b', 'c', 'd'], slist = ['b', 'c'], rlist = ['e']
        输出：['a', 'e', 'd']
    """
    slen = len(slist)
    pos = listFind(alist, slist)
    if pos != -1 and slist != rlist:
        alist = alist[0: pos] + alist[pos+slen:]
        alist[pos: pos] = rlist
        alist = listReplace(alist, slist, rlist)
    return alist

# ----------------------list嵌入list相关函数-------------------------
def ListIntertsection(alist):
    """在alist里找出list各项共同的交集"""
    """ 
        输入：alist = [['a'], ['a', 'b'], ['a', 'c', 'd']]
        输出：True, ['a']
    """
    rbool = False
    rlist = []
    if alist:
        tlist = alist[0]
        for ilist in alist[1:]:
            # 找出tlist和ilist共同的元素
            tlist = list(set(tlist) & set(ilist))
            if not tlist:
                break
        if tlist:
            rbool = True
            rlist = tlist
    return rbool, rlist

# ----------------------list嵌入dict(json格式)相关函数-------------------------
def listAddRank(alist):
    """list嵌入dict(json格式)中对每一行第一项增加rank属性"""
    rlist = []
    for i in range(len(alist)):
        rdict = dict()
        rdict['rank'] = i + 1
        for (k, v) in alist[i].items():
            rdict[k] = v
        rlist.append(rdict)
    return rlist

# ----------------------list嵌入numpy相关函数-------------------------
def listRemoveNpArray(L, arr):
    """删掉list中的np的array"""
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    return L

# ----------------------list与计算相关函数-------------------------
# def dictAccumulate(tlist, nlist):
#     """list元组累加，且保持从大到小排序"""
#     """
#         输入：tlist = [('a', 3), ('b', 2), ('c', 1)]
#              ndict = [('a', 1), ('b', 1), ('c', 1), ('d', 5)]
#         输出：tdict = [('d', 5), ('a', 4), ('b', 3), ('c', 2)]
#     """
#     if tlist:
#         titem = [i[0] for i in tdict]
#         for item in nlist:
#             if item[0] in titem:
#                 tlist[item[1]] =
#             tlist[item] = tlist[item] + v
#     else:
#         tlist = nlist
#     return tlist