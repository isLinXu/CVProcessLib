# -*- coding: utf-8 -*-
'''
@author: linxu
@contact: 17746071609@163.com
@time: 2021-06-05
@desc: 呼吸器识别相关技术
'''
import cv2
import numpy as np

def respirator_color(img):
    """呼吸器识别"""
    src = cv2.resize(img, (250, 450))  # 统一大小
    data = colorRecognition(src)
    print(data)
    # ture:红，false：白
    if data[0] == '红色':
        return True
    else:
        return False

def colorRecognition(img):
    """颜色识别"""
    img2 = img[img.shape[0] * 1 // 5:img.shape[0] * 4 // 5, img.shape[1] * 1 // 5:img.shape[1] * 4 // 5]
    w, h, c = img2.shape
    w_2 = w // 5
    h_2 = h // 5
    total = w_2 * h_2
    HSV_list = []
    rata_rlist = []
    rata_ylist = []
    rata_blist = []
    rata_glist = []
    HSV_total = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    for i in range(5):
        for j in range(5):
            HSV_list.append(HSV_total[i * w_2:(i + 1) * w_2, j * h_2:(j + 1) * h_2])

    # 红色掩膜
    for HSV in HSV_list:
        lower_r1 = np.array([0, 46, 43])
        upper_r1 = np.array([8, 256, 256])
        mask_r1 = cv2.inRange(HSV, lower_r1, upper_r1)
        lower_r1_ref = np.array([0, 0, 225])
        upper_r1_ref = np.array([8, 46, 256])
        mask_r1_ref = cv2.inRange(HSV, lower_r1_ref, upper_r1_ref)
        lower_r2 = np.array([140, 46, 43])
        upper_r2 = np.array([181, 256, 256])
        mask_r2 = cv2.inRange(HSV, lower_r2, upper_r2)
        lower_r2_ref = np.array([140, 0, 225])
        upper_r2_ref = np.array([181, 46, 256])
        mask_r2_ref = cv2.inRange(HSV, lower_r2_ref, upper_r2_ref)
        mask_r = mask_r1 + mask_r2 + mask_r1_ref + mask_r2_ref
        rata_r = np.sum(mask_r == 255) / total
        rata_rlist.append(rata_r)

        # 白色掩膜
        lower_y1 = np.array([15, 20, 43])
        upper_y1 = np.array([50, 256, 256])
        mask_y1 = cv2.inRange(HSV, lower_y1, upper_y1)
        lower_y1_ref = np.array([15, 0, 128])
        upper_y1_ref = np.array([100, 20, 256])
        mask_y1_ref = cv2.inRange(HSV, lower_y1_ref, upper_y1_ref)
        mask_y = mask_y1 + mask_y1_ref
        rata_y = np.sum(mask_y == 255) / total
        rata_ylist.append(rata_y)

        # 蓝色掩膜
        lower_b1 = np.array([93, 46, 43])
        upper_b1 = np.array([110, 256, 256])
        mask_b1 = cv2.inRange(HSV, lower_b1, upper_b1)
        lower_b1_ref = np.array([93, 0, 225])
        upper_b1_ref = np.array([110, 46, 256])
        mask_b1_ref = cv2.inRange(HSV, lower_b1_ref, upper_b1_ref)
        mask_b = mask_b1 + mask_b1_ref
        rata_b = np.sum(mask_b == 255) / total
        rata_blist.append(rata_b)

        # 绿色掩膜
        lower_g1 = np.array([70, 46, 43])
        upper_g1 = np.array([93, 256, 256])
        mask_g1 = cv2.inRange(HSV, lower_g1, upper_g1)
        lower_g1_ref = np.array([70, 0, 225])
        upper_g1_ref = np.array([93, 46, 256])
        mask_g1_ref = cv2.inRange(HSV, lower_g1_ref, upper_g1_ref)
        mask_g = mask_g1 + mask_g1_ref
        rata_g = np.sum(mask_g == 255) / total
        rata_glist.append(rata_g)

    # 进行颜色区域计数，颜色占比超过50%则判定该区域为该颜色。
    num_r = np.sum(np.array(rata_rlist) > 0.65)
    num_y = np.sum(np.array(rata_ylist) > 0.65)
    num_g = np.sum(np.array(rata_glist) > 0.65)
    num_b = np.sum(np.array(rata_blist) > 0.65)

    color_num = [num_r, num_y, num_g, num_b]
    color_names = ['红色', '白色', '绿色', '蓝色']
    color_rgb = ['255,0,0', '255,255,255', '0,255,0', '0,0,255']

    score = max(color_num)
    if score < 5:
        color_result = '白色'
        color_rgb_result = '255,255,255'
        color_proportion_result = score
    else:
        max_location = color_num.index(score)
        color_result = color_names[max_location]
        color_rgb_result = color_rgb[max_location]
        color_proportion_result = score
    data = [color_result, color_proportion_result, color_rgb_result]
    return data


if __name__ == '__main__':
    # path = '/home/linxu/Documents/Work/sources/呼吸器图片/hx1.png'
    path = '/home/linxu/Desktop/呼吸器/14.png'
    img = cv2.imread(path)
    cv2.imshow('img', img)
    cv2.waitKey()
    # 呼吸器识别{红,白}
    img_result = respirator_color(img)
    print(img_result)