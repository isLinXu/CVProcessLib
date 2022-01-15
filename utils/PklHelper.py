# -*- coding:utf-8 -*-

"""
@ Author: LinXu
@ Contact: 17746071609@163.com
@ Date: 2022/01/15 15:37 PM
@ Software: PyCharm
@ File: PklHelper.py
@ Desc: .pkl格式相关支持函数

什么是pkl文件：
1）python中有一种存储方式，可以存储为.pkl文件。
2）该存储方式，可以将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
3）保存方式就是保存到创建的.pkl文件里面。
4）然后需要使用的时候再 open，load。
"""
import pickle
import numpy as np

def save_pkl(np_array,pkl_path):
	with open(pkl_path, 'wb') as f:
		pickle.dump(np_array, f)

def load_pkl(pkl_path):
	with open(pkl_path, 'rb') as f:
		# 读取pkl文件，使用numpy或pickle都可以
		# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
		_pkl = pickle.load(f)
		# 若numpy version>1.14,则需要设置allow_pickle,version<=1.14则不用
		img_data = np.load(pkl_path,allow_pickle=True)
		print(img_data)
		print(_pkl)
	return _pkl

if __name__ == '__main__':
	np_array = [[0,0,0],[1,1,1],[2,2,2]]
	pkl_path = 'array.pkl'
	save_pkl(np_array, pkl_path)
	load_pkl(pkl_path)