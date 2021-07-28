# coding: utf-8
"""
 Computer Vision 2017 (week 3,4): Example of feature matching & homography
   Hiroaki Kawashima <kawashima@i.kyoto-u.ac.jp>

   Ref: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
"""

import numpy as np
import cv2

class ObjectDetector:
    """ 基于特征匹配的实时平面目标检测算法 """
    def __init__(self):
        # 特征点检测-选择不同的特征描述子
        self.feature_detector = cv2.AKAZE_create() # Use AKAZE
        # self.feature_detector = cv2.ORB_create() # Use ORB
        # self.feature_detector = cv2.KAZE_create()# Use KAZE
        # self.feature_detector = cv2.SIFT_create()# Use SIFT
        # self.feature_detector = cv2.BRISK_create()# Use BRISK

        # 摄像头相机参数设置VideoCapture
        self.vidcap = cv2.VideoCapture(0)
        self.vidcap.set(3, 640) # 宽度
        self.vidcap.set(4, 480) # 高度
        self.vidcap.set(5, 15)  # 帧率

        # 通过ROI（感兴趣区域）来注册目标对象
        self.sub_topleft = [100, 220] # [0, 0] # [y,x]100 220
        self.sub_width = 200 #640 200
        self.sub_height = 200 #480 200
        self.sub_bottomright = [self.sub_topleft[0] + self.sub_height - 1,\
        self.sub_topleft[1] + self.sub_width - 1]
        # rect矩形框体
        self.rect_color = (0, 255, 0) # green
        self.rect_thickness = 3
        self.rect_tl_outer_xy = (self.sub_topleft[1] - self.rect_thickness, self.sub_topleft[0] - self.rect_thickness)
        self.rect_br_outer_xy = (self.sub_bottomright[1] + self.rect_thickness, self.sub_bottomright[0] + self.rect_thickness)

        # 特征（描述符）向量距离的阈值
        self.ratio = 0.75
        self.registered = False
        self.min_match_count = 5
        self.show_rectangle = True

    def register(self):
        """注册目标对象"""
        print("\n将目标物体靠近相机.")
        print("确保对象完全覆盖矩形内部（背景不可见）.")
        print("然后，按“r”注册对象.\n")

        while self.vidcap.isOpened():
            ret, frame = self.vidcap.read()
            cv2.rectangle(frame, self.rect_tl_outer_xy, self.rect_br_outer_xy,\
                          self.rect_color, self.rect_thickness)
            cv2.imshow("Registration (press 'r' to register)", frame)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                # 图像切片
                subimg = frame[self.sub_topleft[0]:(self.sub_topleft[0] + self.sub_height),
                               self.sub_topleft[1]:(self.sub_topleft[1] + self.sub_width)]
                self.kp0, self.des0 = self.feature_detector.detectAndCompute(subimg, None)
                self.queryimg = subimg
                self.registered = True
                break

    def detect(self):
        """ 使用特征点查找对象 """
        global mask
        if not self.registered:
            print("Call 'register()' first.")
            return

        print("Start detection...")
        print("按“q”退出.")
        print("按“h”隐藏绿色矩形.\n")

        # 声明一个暴力匹配器Blute-Force (BF) matcher
        bf = cv2.BFMatcher()

        while self.vidcap.isOpened():
            ret, frame = self.vidcap.read()

            # 关键点（kp）检测和计算描述符（des）
            kp, des = self.feature_detector.detectAndCompute(frame, None)

            # 在关键点之间应用knn匹配
            matches = bf.knnMatch(self.des0, des, k=2)

            # 根据阈值筛选关键特征点
            # good = [[m] for m, n in matches if m.distance < self.ratio * n.distance]
            good = []
            for m, n in matches:
                if m.distance < self.ratio * n.distance:
                    good.append([m])
            print('len',len(good))

            contours = []
            # 查找单应性矩阵
            if (len(good) > self.min_match_count) and self.show_rectangle:
                # 建立坐标矩阵
                src_pts = np.float32([self.kp0[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)


                dst_pts = np.float32([kp[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
                # print('dst_pts',dst_pts[0][0])

                # 计算多个二维点对之间的最优单映射变换矩阵 H
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # for contour in good:
                #     contour.append(dst_pts[0][0])
                # print('contours')
                # x,y,w,h = cv2.boundingRect(contours)

                # Assume color camera
                # cv2.imshow('queryimg',self.queryimg)
                h, w, c = self.queryimg.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                # cv2.circle(frame,tuple(dst_pts[0][0]),5, (255,0,0))
                # cv2.circle(frame, tuple(dst_pts[1][0]), 5, (0, 255, 0))
                # cv2.circle(frame, tuple(dst_pts[2][0]), 5, (0, 0, 255))
                # cv2.circle(frame, tuple(dst_pts[3][0]), 5, (255, 255, 0))
                # cv2.imshow('circle', frame)
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

            # 可视化匹配过程
            # 绘画参数
            # draw_params = dict(flags=2)
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),flags=0)
            img = cv2.drawMatchesKnn(self.queryimg, self.kp0, frame, kp, good, None, **draw_params)
            cv2.imshow("Detection (press 'q' to quit)", img)

            key_pressed = cv2.waitKey(1)
            if key_pressed & 0xFF == ord('q'):
                break

            if key_pressed & 0xFF == ord('h'):
                self.show_rectangle = False

    def close(self):
        """ 释放VideoCapture并销毁windows """
        self.vidcap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    obj_detector = ObjectDetector()
    obj_detector.register()
    obj_detector.detect()
    obj_detector.close()
