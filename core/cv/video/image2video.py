# encoding=utf-8
import os
import cv2
import sys
import glob

def image2video_numSort(path='', down=0, up=598, fps=10, size = (864, 504)):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter('output.avi', fourcc, fps, (864, 504))
    for i in range(down, up):
        file_path = path + str(i) + '.png'
        print(file_path)
        frame = cv2.imread(file_path)
        # cv2.imshow('img', img)
        # cv2.waitKey()
        videoWriter.write(frame)
    videoWriter.release()


def image2video_normal(img_rpath='', img_type='png', video_name='output', video_fps=10, video_type='mp4', size = (864, 504)):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # # fourcc = cv2.CV_FOURCC('M','J','P','G')
    video_file = img_rpath + '/' + video_name + '.' + video_type
    # print(video_file)
    videoWriter = cv2.VideoWriter(video_file, fourcc, video_fps, size)
    # 图片路径
    img_list = glob.glob(img_rpath + '/*.' + img_type)
    img_list.sort()
    # img_list.sort(key=lambda x:int(x.split('.')[0]))
    for img_path in sorted(img_list):
        print(img_path)
        frame = cv2.imread(img_path)
        videoWriter.write(frame)
    videoWriter.release()


if __name__ == '__main__':
    img_root = '/home/hxzh02/WORK/Image2Video-master/images/'
    size = (864, 504)
    image2video_normal(img_root)
    # image2video_numSort()
