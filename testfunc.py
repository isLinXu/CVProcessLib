
import os
import cv2

from package.StatusIndicator_Recognition import judge_open_close

if __name__ == '__main__':

    print('hello world')
    file = '/home/linxu/Documents/Work/testpic/'
    respirator_file = '呼吸器/'
    status_file = '状态指示器/'
    index = 0
    png = '.jepg'

    respirator_file_path = file+respirator_file
    status_file_path = file+status_file
    status_file_green_path = status_file_path+'Green/'
    status_file_red_path = status_file_path+'Red/'

    print(status_file_green_path)
    all_count = 0
    true_count = 0
    false_count = 0
    for dirpath, dirnames, filenames in os.walk(status_file_green_path):
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            print(img_path)
            img = cv2.imread(img_path)
            #　识别算法1
            is_true = judge_open_close(img)
            print('is_true',is_true)
            if is_true == True:
                true_count += 1
            elif is_true == False:
                false_count += 1
            all_count += 1

    print('all_count=', all_count)
    print('true_count=', true_count)
    print('false_count=', false_count)
    print('ture_rate=', float(true_count / all_count))
    print('false_rate=', float(false_count / all_count))

    # Green
    # ture_rate= 0.9904761904761905

    # Red
    # false_rate = 0.9262948207171314