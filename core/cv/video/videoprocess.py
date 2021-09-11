'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-06-08 上午09:24
@desc: 视频逐帧存为图片
'''

import cv2
import os
def save_from_video(file_path, save_path):
    # 提取的视频路径
    video_path = os.path.join(file_path)
    times = 0
    # 提取视频的频率，每1帧提取一个
    frameFrequency = 30
    # 输出图片到当前目录文件夹下
    outPutDirName = save_path
    if not os.path.exists(outPutDirName):
        # 如果文件目录不存在则创建目录
        os.makedirs(outPutDirName)
    camera = cv2.VideoCapture(video_path)
    while True:
        times += 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            cv2.imwrite(outPutDirName + str(times) + '.jpg', image)
            print(outPutDirName + str(times) + '.jpg')
    print('图片提取结束')
    camera.release()

if __name__ == '__main__':
    # file_path = '/home/linxu/Documents/Work/sources/交通灯识别（重视红绿灯）/交通灯识别（重视红绿灯）/红绿灯最终测试视频.mp4'
    # file_path = '/home/linxu/Documents/Work/sources/交通灯识别（重视红绿灯）/交通灯识别（重视红绿灯）/test3.avi'
    file_path = '/home/hxzh02/文档/【原创】4K·高压电线铁塔.mp4'
    save_path = '/home/hxzh02/文档/【原创】4K·高压电线铁塔/'
    save_from_video(file_path,save_path)
