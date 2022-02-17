# labelVideo.py
import os
import cv2

cap = cv2.VideoCapture(0)

# 根目录
basedir = "d:/video"
# 标签
label = "click"
# 序号
index = 0
# size 图片大小
size = (224, 224)
# 开启
on = False

nowPath = "{}/{}".format(basedir, label)
if not os.path.exists(nowPath):
    os.makedirs(nowPath + "/")
print(nowPath)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
videowrite = cv2.VideoWriter(os.path.join(nowPath, "{}.avi".format(index)), fourcc, 10, size)
while True:
    cat, frame = cap.read()
    cv2.imshow("{}_{}".format(label, index), frame)
    key = cv2.waitKey(0)

    if key & 0xFF == ord("q"):
        break

    elif key & 0xFF == ord("s"):
        index += 1
        videowrite.release()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videowrite = cv2.VideoWriter(os.path.join(nowPath, "{}.avi".format(index)), fourcc, 10, size)
        print(os.path.join(nowPath, "{}.avi".format(index)))
        on = True  # 表示可写入文件

    elif key & 0xFF == ord("d"):
        on = False

    elif key & 0xFF == ord("a"):
        if on:
            resize = cv2.resize(frame, size)
            videowrite.write(resize)
    else:
        print(key)
cap.release()
videowrite.release()
cv2.destroyAllWindows()
