import cv2
import os
import multiprocessing
from tqdm import tqdm
from datetime import datetime, time

def extract_frames(video_path, output_path, frame_interval):
    video_capture = cv2.VideoCapture(video_path)
    count = 0

    if not video_capture.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    os.makedirs(output_path, exist_ok=True)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_rate = int(fps * frame_interval)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=int(total_frames/frame_rate), desc="抽帧进度")

    while video_capture.isOpened():
        success, frame = video_capture.read()

        if success:
            if count % frame_rate == 0:
                video_dir = output_path.split("/")[-1]
                video_name = os.path.basename(video_path).split('.')[0]
                output_filename = os.path.join(output_path, f"{video_dir}_frame_{count}.jpg")
                cv2.imwrite(output_filename, frame)
                progress_bar.update(1)

            count += 1
        else:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    progress_bar.close()

def extract_frames_from_directory(input_dir, output_dir, frame_interval=15):
    time_ranges = [(time(7, 0), time(10, 0)), (time(12, 0), time(14, 0)), (time(17, 0), time(20, 0))]

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.mp4'):
                video_time = datetime.strptime(filename.split('_')[2].split('.')[0], "%Y%m%d%H%M%S").time()
                if any(start <= video_time <= end for start, end in time_ranges):
                    file_path = os.path.join(root, filename)
                    video_name = filename[:-4]
                    output_path = os.path.join(output_dir, video_name)
                    os.makedirs(output_path, exist_ok=True)
                    p = multiprocessing.Process(target=extract_frames, args=(file_path, output_path, frame_interval))
                    p.start()
                    print(f"开始处理视频文件: {file_path}")

input_directories = [
    "/path/to/input",
]

output_base_dir = "/path/output"
frame_interval = 15

for input_dir in input_directories:
    output_dir = os.path.join(output_base_dir, os.path.basename(input_dir))
    extract_frames_from_directory(input_dir, output_dir, frame_interval)