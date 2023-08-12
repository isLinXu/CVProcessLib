
import cv2
import os
import multiprocessing
from tqdm import tqdm


def extract_frames(video_path, output_path, start_time, end_time, frame_rate=1):
    '''
    从视频中抽取帧
    :param video_path: 视频路径
    :param output_path: 输出路径
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param frame_rate: 抽帧频率
    '''
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
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)

    progress_bar = tqdm(total=int((end_frame - start_frame) / frame_rate), desc="抽帧进度")
    # print("start_frame: ", start_frame," end_frame: ", end_frame)
    while True:
        # 读取一帧视频
        success, frame = video_capture.read()

        if success:

            print("count:", count, "start_frame: ", start_frame, " end_frame: ", end_frame)
            # 根据帧率控制抽帧
            if count >= start_frame and count <= end_frame and count % frame_rate == 0:
                # 拼接输出文件路径
                video_dir = output_path.split("/")[-1]
                # print("video_dir:", video_dir)
                video_name = os.path.basename(video_path).split('.')[0]
                output_filename = os.path.join(output_path, f"{video_name}_frame_{count}.jpg")
                # 保存帧为图片
                cv2.imwrite(output_filename, frame)
                progress_bar.update(1)

            count += 1

            if count > end_frame:
                break
        else:
            break

    # 释放视频文件和OpenCV窗口
    video_capture.release()
    cv2.destroyAllWindows()
    progress_bar.close()


def extract_frames_from_directory(input_dir, output_dir, start_time, end_time, frame_rate=1):
    '''
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param start_time: 视频播放开始时间
    :param end_time: 视频播放结束时间
    :param frame_rate: 抽帧帧率
    :return:
    '''

    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
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
                output_path = os.path.join(output_dir, video_name)
                os.makedirs(output_path, exist_ok=True)
                # 调用抽帧函数
                p = multiprocessing.Process(target=extract_frames,
                                            args=(file_path, output_path, start_seconds, end_seconds, frame_rate))
                p.start()


def time_to_seconds(time_str):
    # 将时间字符串转换为秒数
    time_parts = time_str.split(':')
    minutes = int(time_parts[0])
    seconds = int(time_parts[1])
    total_seconds = minutes * 60 + seconds
    return total_seconds


def get_video_duration(video_path):
    # 打开视频文件
    video_capture = cv2.VideoCapture(video_path)

    # 检查视频文件是否成功打开
    if video_capture.isOpened() == False:
        print("无法打开视频文件")
        return None

    # 获取视频总帧数和帧率
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # 计算视频播放时间
    duration_seconds = total_frames / fps
    duration_minutes = int(duration_seconds / 60)
    duration_seconds = int(duration_seconds % 60)

    # 释放视频文件和OpenCV窗口
    video_capture.release()

    return duration_minutes, duration_seconds



if __name__ == '__main__':
    # 视频路径
    video_path = '/Users/gatilin/youtu-work/34020000001320000003_2023-08-11_05-19-27.mp4'

    # 输出目录
    output_path = '/Users/gatilin/youtu-work/34020000001320000003_2023-08-11_05-19-27'

    # 获取视频播放时间段区间
    duration_minutes, duration_seconds = get_video_duration(video_path)
    # start_frame, end_frame = get_frame_range(video_path, start_time, end_time)

    print(f"视频总时长为：{duration_minutes} 分钟 {duration_seconds} 秒")

    # 视频播放开始时间&结束时间-格式：分钟:秒钟
    start_time = '8:35'
    end_time = '11:35'

    # 抽帧帧率
    frame_rate = 3

    extract_frames(video_path, output_path, start_time, end_time, frame_rate=1)