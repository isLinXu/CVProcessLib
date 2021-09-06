import cv2


from core.cv.skeleton.skeletonize import scikit_skeletonize
from core.cv.line.fast_line_detection import FLD

if __name__ == '__main__':
    file_path = '/home/linxu/Desktop/无人机巡检项目/输电杆塔照片素材/输电杆塔照片素材/杆塔倒塌/1.JPG'
    img = cv2.imread(file_path)
    img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    skeleton = scikit_skeletonize(img)
    cv2.imshow('img',img)
    cv2.imshow('skeleton',skeleton)
    cv2.waitKey()

    skeleton = cv2.cvtColor(skeleton,cv2.COLOR_GRAY2RGB)
    xha, yha, img = FLD(skeleton)
    # cv2.imshow('fld', img)
    # cv2.waitKey()