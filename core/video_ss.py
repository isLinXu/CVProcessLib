
from vidstab import VidStab

stabilizer = VidStab()

input_path = '/home/linxu/Desktop/视频防抖/video.mp4'

stabilizer.stabilize(input_path=input_path,
                     output_path='stable_webcam.avi',
                     max_frames=1000,
                     playback=True)

