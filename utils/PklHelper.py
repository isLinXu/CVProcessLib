# -*- coding:utf-8 -*-

"""
@ Author: LinXu
@ Contact: 17746071609@163.com
@ Date: 2022/01/15 15:37 PM
@ Software: PyCharm
@ File: PklHelper.py
@ Desc: .pkl格式相关支持函数
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