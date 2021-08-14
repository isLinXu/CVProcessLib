import cv2
import os

def change_all_videos_name(file_dir):
    """
    修改获得到的视频名称，将杂乱的名称从00001开始计数；
    :param path:
    :return:
    """
    videos_name_list = sorted(os.listdir(file_dir))
    file_dtype = ".jpg"
    video_num = 0

    for video_name in videos_name_list:
        video_path = os.path.join(file_dir, video_name)
        video_new_name = (5 - len(str(video_num))) * "0" + str(video_num) + file_dtype
        video_new_path = os.path.join(file_dir, video_new_name)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)

        os.rename(video_path, video_new_path)
        video_num += 1

    return 0


def images_to_videos(path_dir, output_path, position, fps=25):
    """
    对连续帧的图片截取特定位置并转化为视频文件
    :param path_dir: 图片的文件夹 
    :param output_path: 视频输出的路径
    :param position: 图片截取的位置[y1, y2, x1, x2]
    :param fps: 帧率
    :return: 
    """""
    y1, y2, x1, x2 = position

    # fps = 25  # 获取视频的帧率
    size = (x2 - x1, y2 - y1)  # 获取视频的大小
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')  # 要保存的视频格式
    output_viedo = cv2.VideoWriter()
    video_save_path = output_path
    output_viedo.open(video_save_path, fourcc, fps, size, True)

    path_image_list = os.listdir(path_dir)
    for path_image in path_image_list:
        path_now = os.path.join(path_dir, path_image)
        image_now = cv2.imread(path_now)
        image_now_cut = image_now[y1:y2, x1:x2]
        output_viedo.write(image_now_cut)

    output_viedo.release()  # 释放

    return 0


def distinguish_and_images_to_video(images_dir_path):
    """
    将名称连续的图片合并成为视频
    :param images_dir_path:
    :return:
    """

    image_name_list = sorted(os.listdir(images_dir_path))
    image_name_num_next = None
    save_num = 0
    output_viedo = None

    for image_name in image_name_list:
        # 读取图片并以及相关信息
        image_path = os.path.join(images_dir_path, image_name)
        image_name_num = int(image_name[:-4])
        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        if save_num == 0:
            # 刚进入循环需要第一次进行赋值
            present_num_str = (5 - len(str(save_num))) * "0" + str(save_num)
            new_video_path = os.path.join(images_dir_path, present_num_str) + ".mp4"
            fps = 25
            fourcc = cv2.VideoWriter_fourcc(*'mpeg')  # 要保存的视频格式
            output_viedo = cv2.VideoWriter(new_video_path, fourcc, fps, (width, height))
            output_viedo.write(image)
            save_num += 1
            image_name_num_next = image_name_num

        if image_name_num == image_name_num_next:
            image_name_num_next += 1

            output_viedo.write(image)

        else:  # 两张图片若是序号不连接，则说明进入新一个视频
            image_name_num_next = image_name_num + 1

            present_num_str = (5 - len(str(save_num))) * "0" + str(save_num)
            new_video_path = os.path.join(images_dir_path, present_num_str) + ".mp4"
            output_viedo = cv2.VideoWriter(new_video_path, fourcc, fps, (width, height))
            save_num += 1
            output_viedo.write(image)
    return 0


def video_to_list(cap):
    """
    将视频流的图片转成图片数组
    :param cap:视频流
    :return:
    """
    images_list = []
    while True:
        grabbed, image_get_from_video = cap.read()  # 逐帧采集视频流
        if not grabbed:
            break

        # cv2.imshow("image_get_from_video", image_get_from_video)
        images_list.append(image_get_from_video)
        if cv2.waitKey(40) & 0xFF == ord('q'):  # 等候40ms,播放下一帧，或者按q键退出
            break
    return images_list