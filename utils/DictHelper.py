# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: whang@ikeeko.com
@time: 2018-06-04 下午18:12
@desc: dict函数辅助
'''

# ----------------------dict相关函数-------------------------
def dictAccumulate(tdict, ndict):
    """dict累加"""
    """ 
        输入：tdict = {'a':1, 'b':2, 'c':3}
             ndict = {'a':1, 'b':1, 'c':1, 'd':5}
        输出：tdict = {'a':2, 'b':3, 'c':4, 'd':5}
    """
    if tdict:
        for k, v in ndict.items():
            if k in tdict:
                tdict[k] = tdict[k] + v
            else:
                tdict[k] = v
    else:
        tdict = ndict
    return tdict