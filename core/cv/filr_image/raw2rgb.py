#!/usr/bin/python

__author__ = "shondll"
__license__ = "GPL"
__version__ = "1.0."
__status__ = "Prototype"
'''
Usage for 10bit data in LSB format and resolution 3840x2160 is like this:
./raw2rgb.py file.bin 3840 2160 --stride 7680
usage: RAW2RGB Converter [-h] [--stride STRIDE] [--frame FRAME] [--version]
                         filename width height
Convert Raw images into RGB format

positional arguments:
  filename
  width
  height

optional arguments:
  -h, --help       show this help message and exit
  --stride STRIDE  stride of the image (default: width of the image
  --frame FRAME    frame number to grab (default: index 0)
  --version        show program's version number and exit
'''
import argparse
from PIL import Image
import sys
import struct
import os
import math

DEFAULT_STRIDE = -1
DEFAULT_FRAME = 0

parser = argparse.ArgumentParser(prog='RAW2RGB Converter', description='Convert Raw images into RGB format')
parser.add_argument('filename', type=str)
parser.add_argument('width', type=int)
parser.add_argument('height', type=int)
parser.add_argument('--stride', default=DEFAULT_STRIDE, type=int, help='stride of the image (default: width of the image')
parser.add_argument('--frame', default=DEFAULT_FRAME, type=int, help='frame number to grab (default: index 0)')
parser.add_argument('--version',action='version', version='%(prog)s 1.0')

class Converter(object):
	"""Base class that should define interface for all conversions"""
	def __init__(self, filename, width, height, stride, frame):
		self.filename = filename
		self.width = width
		self.height = height
		self.stride = stride
		self.frame = frame
	#constructor#

	def Convert():
		raise NotImplementedError( "Should have implemented this!" )
	#convert#
#Converter#



class YUY2Converter(Converter):
	"""This class converts NV12 files into RGB"""
	def __init__(self, filename, width, height, stride, frame):
		super(YUY2Converter, self).__init__(filename, width, height, stride, frame)
	#constructor#

	def getPx(self, arr, x, y):
		if x < 0:
			x = 0
		if y < 0:
			y = 0

		if x >= self.width:
			x = self.width - 1
		if y >= self.height:
			y = self.height - 1

		return struct.unpack_from('<h', arr, self.stride * y + x * 2)[0]
		
	def ConvertBayerSmart(self):	
		f = open(self.filename, "rb")

		converted_image_filename = self.filename.split('.')[0] + ".smart.bmp"
		converted_image = Image.new("RGB", (self.width, self.height) )
		pixels = converted_image.load()

		f.seek(0)
		arr = bytearray(f.read())
		
		for y in range(0, self.height):
			for x in range(0, self.width):
				if y % 2 == 0: 
					# even line
					if x % 2 == 0:
						# we have B, calc R, G
						b = self.getPx(arr, x, y)
						r = (self.getPx(arr, x-1, y-1) + self.getPx(arr, x-1, y+1) + self.getPx(arr, x+1, y-1) + self.getPx(arr, x+1, y+1)) / 4
						g = (self.getPx(arr, x, y-1) + self.getPx(arr, x, y+1) + self.getPx(arr, x-1, y) + self.getPx(arr, x+1, y)) / 4
					else:
						# we have G, calc R, B, B on the same line
						g = self.getPx(arr, x, y)
						r = (self.getPx(arr, x, y-1) + self.getPx(arr, x, y+1)) / 2
						b = (self.getPx(arr, x-1, y) + self.getPx(arr, x+1, y)) / 2
				else:
					#odd line
					if x % 2 == 0:
						# we have G, calc R, B, R on the same line
						g = self.getPx(arr, x, y)
						r = (self.getPx(arr, x-1, y) + self.getPx(arr, x+1, y)) / 2
						b = (self.getPx(arr, x, y-1) + self.getPx(arr, x, y+1)) / 2
					else:
						# we have R, calc R, G
						r = self.getPx(arr, x, y)
						b = (self.getPx(arr, x-1, y-1) + self.getPx(arr, x-1, y+1) + self.getPx(arr, x+1, y-1) + self.getPx(arr, x+1, y+1)) / 4
						g = (self.getPx(arr, x, y-1) + self.getPx(arr, x, y+1) + self.getPx(arr, x-1, y) + self.getPx(arr, x+1, y)) / 4

				denom = 4
				pixels[x,y] = int(r)/denom, int(g)/denom, int(b)/denom

		converted_image.save(converted_image_filename)

	def ConvertBayer(self):
		f = open(self.filename, "rb")

		converted_image_filename = self.filename.split('.')[0] + ".bayer.bmp"
		converted_image = Image.new("RGB", (self.width/2, self.height/2) )
		pixels = converted_image.load()

		frame_start = 0

		f.seek(frame_start);
		for j in range(0, self.height/2):
			row0 = bytearray(f.read(self.width * 2))
			row1 = bytearray(f.read(self.width * 2))
			for i in range(0, self.width/2, 2):
				[b, g] = struct.unpack_from('<hh', row0, i * 4)
				[g1, r] = struct.unpack_from('<hh', row1, i * 4)

				pixels[i,j] = int(r/4), int((g+g1)/8), int(b/4)

		converted_image.save(converted_image_filename)



def main():
	args = parser.parse_args()

	if args.stride == DEFAULT_STRIDE:
		args.stride = args.width

	if args.frame < 0:
		args.frame = 0

	converter = YUY2Converter(args.filename, args.width, args.height, args.stride, args.frame)
	converter.ConvertBayerSmart()

#main function#
if __name__ == "__main__":
	main()
