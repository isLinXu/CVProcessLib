from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
import ntpath

# For windows, the below if condition is must.
if __name__ == "__main__":
    # 包含所有视频的目录路径。
    # /home/hxzh02/文档/导线视频测试/DJI_20211228111741_0039_S.MP4
    dir_path = '/home/hxzh02/文档/导线视频测试'
    # 输出视频
    folder_loc = "/home/hxzh02/文档/导线视频测试/output/"
    # videos_dir_path = ""

    # instantiate the video class
    vd = Video()

    # number of key-frame images to be extracted
    # 要提取的关键帧数
    no_of_frames_to_return = 100

    # Input Video directory path
    # All .mp4 and .mov files inside this directory will be used for keyframe extraction)
    # videos_dir_path = os.path.join(".", "tests", "data")
    videos_dir_path = '/home/hxzh02/文档/导线视频测试/DJI_20211228111741_0039_S.mp4'
    # diskwriter = KeyFrameDiskWriter(location="selectedframes")

    diskwriter = KeyFrameDiskWriter(location=folder_loc)

    vd.extract_keyframes_from_videos_dir(no_of_frames=no_of_frames_to_return, dir_path=dir_path, writer=diskwriter)

    vd.extract_keyframes_from_videos_dir(
        no_of_frames=no_of_frames_to_return, dir_path=videos_dir_path,
        writer=diskwriter
    )
