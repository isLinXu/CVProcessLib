# -*-coding:utf-8-*-

'''使用walk方法递归遍历目录文件，walk方法会返回一个三元组，分别是root、dirs和files。
其中root是当前正在遍历的目录路径；dirs是一个列表，包含当前正在遍历的目录下所有的子目录名称，不包含该目录下的文件；
files也是一个列表，包含当前正在遍历的目录下所有的文件，但不包含子目录。
'''
import os
from PIL import Image
import cv2

def os_mkdir(path):
    '''
    目录创建/操作函数
    :param path:
    :return:
    '''
    # 去除首位空格
    path = path.strip()
    # 去除尾部/符号
    path = path.rstrip("/")
    # 判断路径是否存在
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def images_Normalization(path, img_show=False):
    for root, dirs, files in os.walk(path):
        print('################################################################')
        for name in files:
            if len(dirs) == 0:
                fname = os.path.join(root, name)
                print('fname', fname)
                print('name', name)
                # 处理原图img
                src = cv2.imread(fname)
                src = cv2.resize(src, (640, 480))
                if img_show:
                    cv2.imshow('src_img', src)
                    k = cv2.waitKey() & 0xff
                    if k == 27: return 0
                # 创建src目录并存储图片
                src_dir = root + '/src/'
                src_path = root + '/src/' + name
                print('src_dir', src_dir)
                print('src_path', src_path)
                os_mkdir(src_dir)
                cv2.imwrite(src_path, src)


# 图片采集数据清洗
def dataWash(path, r_tag=False, w_min=300, w_max=1000, h_min=300, h_max=1000):
    '''
    遍历指定文件夹中所有文件
    :param path:
    :return:
    '''
    global timg, img, w, h
    print('开始遍历......')
    for root, dirs, files in os.walk(path):
        ratelist = []
        timglist = []
        idict = {}
        # 第一次清洗,删除文件出错或打不开的图片文件
        print('#######################第一次清洗-开始#######################')
        for name in files:
            if len(dirs) != 0:
                fname = os.path.join(root, name)
                try:
                    timg = os.path.join(root, name)
                    img = Image.open((timg))
                    # 图像文件长与宽
                    w = img.width
                    h = img.height
                    rate = w / h
                    print(fname, 'w=', w, 'h=', h, 'rate=', rate)
                    ratelist.append(rate)
                    timglist.append(timg)

                    # 剔除图片
                    if (r_tag):
                        if w in range(w_min, w_max) and h in range(h_min, h_max):
                            os.remove(timg)
                            print('删除图片')
                            print(timg)
                            pass

                    # 显示图片并进行调整大小
                    src = cv2.imread(fname)
                    cv2.namedWindow('src', cv2.WINDOW_AUTOSIZE)
                    # src = cv2.resize(src, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
                    src = cv2.resize(src, (640, 480))
                    cv2.imshow('src', src)

                    k = cv2.waitKey(0)
                    if k == 27: return 0


                except:
                    print('删除图片', timg)
                    os.remove(timg)  # 删除打不开的文件

                img.close()
            print('#######################第一次清洗-结束#######################')
            # print(ratelist)
            ulists = list(set(ratelist))
            # print(ulists)


# 图片采集数据清洗
def dataWash_1(path, r_tag=False, w_min=300, w_max=1000, h_min=300, h_max=1000):
    '''
    遍历指定文件夹中所有文件
    :param path:
    :return:
    '''
    global timg, img, w, h
    print('开始遍历......')
    for root, dirs, files in os.walk(path):
        print('path', path)
        # 第一次清洗,删除文件出错或打不开的图片文件
        print('#######################第一次清洗-开始#######################')
        for name in files:
            fname = os.path.join(root, name)
            print('fname', fname)
            if len(dirs) == 0:
                try:
                    timg = os.path.join(root, name)
                    img = Image.open((timg))
                    # 图像文件长与宽
                    w = img.width
                    h = img.height
                    rate = w / h
                    print(fname, 'w=', w, 'h=', h, 'rate=', rate)

                    # 剔除图片
                    if (r_tag):
                        if w in range(w_min, w_max) and h in range(h_min, h_max):
                            os.remove(timg)
                            print('删除图片')
                            print(timg)
                            pass

                    # 显示图片并进行调整大小
                    src = cv2.imread(fname)
                    cv2.namedWindow('src', cv2.WINDOW_AUTOSIZE)
                    # src = cv2.resize(src, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
                    src = cv2.resize(src, (640, 480))
                    cv2.imshow('src', src)

                    # k = cv2.waitKey(0)
                    # if k == 27: return 0
                    key = cv2.waitKey(0)
                    if key == 27:  # 按esc键退出
                        print('esc break...')
                        cv2.destroyAllWindows()
                        break
                    if key == ord(' '):  # 按空格键删除当前图片
                        print('删除当前图片成功')
                        os.remove(timg)
                    if key == ord('s'):
                        # 创建src目录并存储图片
                        print('保存当前图片到src目录下')
                        src_dir = root + '/src/'
                        src_path = root + '/src/' + name
                        print('src_dir', src_dir)
                        print('src_path', src_path)
                        os_mkdir(src_dir)
                        cv2.imwrite(src_path, src)

                except:
                    print('删除当前图片成功', timg)
                    os.remove(timg)

                img.close()
        print('#######################第一次清洗-结束#######################')



if __name__ == '__main__':
    # path = '/home/hxzh02/文档/defectDetect/金属锈蚀(复件)'
    # path = '/home/hxzh02/文档/PlanetDataset/电力输电塔架'
    # path = '/home/hxzh02/文档/PlanetDataset/输电塔'
    # path = '/home/hxzh02/文档/PlanetDataset/输电塔架'
    # path = '/home/hxzh02/文档/PlanetDataset/输电铁塔'
    # path = '/home/hxzh02/文档/PlanetDataset/猫头塔'
    # path = '/home/hxzh02/文档/PlanetDataset/干字塔'

    # path = '/home/hxzh02/文档/PlanetDataset/src/双回路塔'
    # path = '/home/hxzh02/文档/PlanetDataset/src/电力输电塔架'
    # path = '/home/hxzh02/文档/PlanetDataset/src/单回路塔'
    # path = '/home/hxzh02/文档/PlanetDataset/src/输电塔'
    # path = '/home/hxzh02/文档/PlanetDataset/src/输电塔架'
    # path = '/home/hxzh02/文档/PlanetDataset/src/干字塔'
    # path = '/home/hxzh02/文档/PlanetDataset/src/官帽塔'
    # path = '/home/hxzh02/文档/PlanetDataset/src/酒杯塔'

    path = '/home/hxzh02/文档/PlanetDataset/src/电网铁塔'
    dataWash_1(path)
    # images_Normalization(path)

    # path = '/home/hxzh02/桌面/杆塔倒塌-负样本/'
    # path = '/home/hxzh02/桌面/杆塔图片汇总/'
    # images_Normalization(path, False)
