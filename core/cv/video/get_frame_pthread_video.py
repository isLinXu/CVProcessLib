import cv2
import os
import multiprocessing
from tqdm import tqdm


def extract_frames(video_path, output_path, frame_rate=1):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)
    count = 0

    # 检查视频文件是否成功打开
    if video_capture.isOpened() == False:
        print("无法打开视频文件")
        return

    # 确保输出路径存在
    os.makedirs(output_path, exist_ok=True)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=int(total_frames / frame_rate), desc="抽帧进度")

    while video_capture.isOpened():
        # 读取一帧视频
        success, frame = video_capture.read()

        if success:
            # 根据帧率控制抽帧
            if count % frame_rate == 0:
                # 拼接输出文件路径
                # output_filename = os.path.join(output_path, f"frame_{count}.jpg")
                # print("output_path:", output_path.split("/")[-1])
                # video_dir = video_path.split("/")[-2]

                video_dir = output_path.split("/")[-1]
                # print("video_dir:", video_dir)
                # 获取视频名称
                video_name = os.path.basename(video_path).split('.')[0]
                # 拼接输出文件路径
                output_filename = os.path.join(output_path, f"{video_dir}_frame_{count}.jpg")
                # 保存帧为图片
                cv2.imwrite(output_filename, frame)
                # print(f"save {count} frame")
                progress_bar.update(1)

            count += 1
        else:
            break
    # 释放视频文件和OpenCV窗口
    video_capture.release()
    cv2.destroyAllWindows()
    progress_bar.close()


def extract_frames_from_directory(input_dir, output_dir, frame_rate=1):
    # 遍历输入目录中的所有文件夹和文件
    for root, dirs, files in os.walk(input_dir):
        # 遍历当前目录下的所有文件
        for filename in files:
            # 获取文件路径
            file_path = os.path.join(root, filename)
            # 只处理mp4文件
            if filename.endswith('.mp4'):
                # 创建与视频文件同名的目录
                video_name = filename[:-4]
                # video_name = root.split("/")[-1]
                print("video_name:", video_name)
                # print("file_path:", file_path)
                # file_dir = file_path.split("/")[-2]

                # print("file_dir:", file_dir)
                # 去掉文件后缀
                output_path = os.path.join(output_dir, video_name)
                os.makedirs(output_path, exist_ok=True)
                # # 调用抽帧函数
                p = multiprocessing.Process(target=extract_frames, args=(file_path, output_path, frame_rate))
                p.start()


if __name__ == '__main__':
    # 输入目录
    input_dir = 'video/'

    # 输出目录
    output_dir = 'video_frame'

    frame_rate = 3

    extract_frames_from_directory(input_dir, output_dir, frame_rate)
