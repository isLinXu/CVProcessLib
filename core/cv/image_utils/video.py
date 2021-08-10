'''
视频相关

Author: alex
Created Time: 2020年11月03日 星期二 16时59分44秒
'''
import skvideo.io


def get_video_rotate(video_path):
    """获取视频旋转角度
    注意：手机拍摄的视频，其角度可能需要进行旋转
    """
    metadata = skvideo.io.ffprobe(video_path)
    d = metadata['video'].get('tag')[0]
    if d.setdefault('@key') == 'rotate':     # 获取视频自选择角度
        return 360-int(d.setdefault('@value'))
    return 0
