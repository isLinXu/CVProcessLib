"""
该版本由于图片获取方式不统一，增加了一些不必要的判断
"""
import numpy as np
import cv2
import cv2.aruco as aruco
import os

class Find(object):

    def __init__(self, is_save=False, save_path=None):
        super(Find, self).__init__()

        self.is_save = is_save
        self.save_path = save_path

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        self.parameters = aruco.DetectorParameters_create()

        if self.is_save:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)

    def dlq_crop(self, img, ks=None, save_name=None, dlq_type=None):

        if ks is not None:
            img_ = cv2.GaussianBlur(img, ks, 0)
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None and len(corners) != 0:

            x_min = corners[0][0, :, 0].min()
            y_min = corners[0][0, :, 1].min()
            x_max = corners[0][0, :, 0].max()
            y_max = corners[0][0, :, 1].max()

            if dlq_type == 'ABB-SACE-Emax':
                lower = y_min
                upper = y_min - (y_max - y_min) * 1.1
                left = x_min - (x_max - x_min) * 1.5
                right = x_max + (x_max - x_min) * 1.5

            elif dlq_type == 'ABB-SACE-Emax-X1':
                lower = y_max + (y_max - y_min) * 0.8
                upper = y_max
                left = x_min - (x_max - x_min) * 1.5
                right = x_max + (x_max - x_min) * 1.5
            else:
                return None

            out = img[int(upper):int(lower), int(left):int(right)]

            if out.shape[0] > 0 and out.shape[1] > 0:
                if self.save_path and self.is_save and save_name:
                    image_save_path = self.save_path + save_name
                    cv2.imwrite(image_save_path, out)
                return out
            else:
                return None

        else:
            return None

    def dlq_state(self, img, dlq_type=None, state_thresh=0.0, eps=1e-7):

        img_left = img[0:img.shape[0], 0:int(img.shape[1]*0.5)]
        img_right = img[:, int(img.shape[1]*0.5):]

        if dlq_type == 'ABB-SACE-Emax':

            img_blur = cv2.GaussianBlur(img_left, (7, 7), 0)
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=35, minRadius=int(img_left.shape[0] / 9),
                                       maxRadius=int(img_left.shape[0]))
            if circles is not None:
                # print(img_name)
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # 画出来圆的边界
                    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # 画出来圆心
                    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 1)
            cv2.imshow('1', img)
            cv2.waitKeyEx(0)
            cv2.destroyAllWindows()

            mask_right = self.hsv_mask(img_right, [[15, 50, 50], [35, 255, 255], [0, 0, 0], [0, 0, 0]], 21)
            if circles is not None:
                if ((mask_right > 0).sum() + eps) / mask_right.shape[0] / mask_right.shape[1] > state_thresh + eps:

                    return [False, True]

                else:
                    return [False, False]
            else:

                if ((mask_right > 0).sum() + eps) / mask_right.shape[0] / mask_right.shape[1] > state_thresh + eps:

                    return [True, True]

                else:
                    return [True, False]

        elif dlq_type == 'ABB-SACE-Emax-X1':
            img = img[:, int(img.shape[1] / 2):int(img.shape[1])]
            img_blur = cv2.GaussianBlur(img, (7, 7), 0)
            cv2.imshow('1', img_blur)
            cv2.waitKeyEx(0)
            cv2.destroyAllWindows()
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)

            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=35, minRadius=0,
                                       maxRadius=0)

            if circles is not None:
                # print(img_name)
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # 画出来圆的边界
                    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # 画出来圆心
                    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 1)
                    #print(img_name, 'off')
            cv2.imshow('1', img)
            cv2.waitKeyEx(0)
            cv2.destroyAllWindows()

            mask_left = self.hsv_mask(img_left, [[15, 50, 50], [35, 255, 255], [0, 0, 0], [0, 0, 0]], 21)

            if circles is not None:
                if ((mask_left > 0).sum() + eps) / mask_left.shape[0] / mask_left.shape[1] > state_thresh + eps:

                    return [False, True]

                else:
                    return [False, False]
            else:

                if ((mask_left > 0).sum() + eps) / mask_left.shape[0] / mask_left.shape[1] > state_thresh + eps:

                    return [True, True]

                else:
                    return [True, False]


    def hsv_mask(self, img, h_range_list, ks):

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_01 = np.array(h_range_list[0])
        upper_01 = np.array(h_range_list[1])
        lower_02 = np.array(h_range_list[2])
        upper_02 = np.array(h_range_list[3])
        mask_01 = cv2.inRange(img_hsv, lower_01, upper_01)
        mask_02 = cv2.inRange(img_hsv, lower_02, upper_02)
        mask_out = mask_01 | mask_02
        kernel = np.ones((ks, ks), np.uint8)
        mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_OPEN, kernel)

        return mask_out


if __name__ == '__main__':

    images_path = r'data_2'
    dlq_type = 'ABB-SACE-Emax-X1'
    Emax_ks = (31, 31)
    Emax_X1_ks = (17, 17)

    if dlq_type == 'ABB-SACE-Emax-X1':
        ks = Emax_X1_ks
    elif dlq_type == 'ABB-SACE-Emax':
        ks = Emax_ks

    images = os.listdir(images_path)
    DFind = Find(is_save=True, save_path='./crop/')

    for i, image_name in enumerate(images):
        image = cv2.imread(os.path.join(images_path, image_name))
        crop_img = DFind.dlq_crop(image, ks, image_name, dlq_type)  # 31 11
        if crop_img is not None:

            temp_state = DFind.dlq_state(crop_img, dlq_type)

            image = cv2.putText(image, "({},{})".format(temp_state[0], temp_state[1]), (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)

            image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

            cv2.imshow('temp', image)
            print(image_name, 'state:', temp_state)
            cv2.waitKey()
            cv2.destroyAllWindows()

        else:
            print(image_name)

