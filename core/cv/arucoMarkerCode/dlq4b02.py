import numpy as np
import cv2
import cv2.aruco as aruco


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

            point1 = (corners[0][0][0][0], corners[0][0][0][1])
            point2 = (corners[0][0][1][0], corners[0][0][1][1])
            point3 = (corners[0][0][2][0], corners[0][0][2][1])
            point4 = (corners[0][0][3][0], corners[0][0][3][1])

            x_min = min(point1[0], min(point2[0], min(point3[0], point4[0])))
            y_min = min(point1[1], min(point2[1], min(point3[1], point4[1])))
            x_max = max(point1[0], max(point2[0], max(point3[0], point4[0])))
            y_max = max(point1[1], max(point2[1], max(point3[1], point4[1])))

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

        img_left, img_right = img[:, :int(img.shape[1]*0.5)], img[:, int(img.shape[1]*0.5):]

        if dlq_type == 'ABB-SACE-Emax':

            img_blur = cv2.GaussianBlur(img_left, (3, 3), 0)
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                       param1=50, param2=35, minRadius=int(img_left.shape[0] / 9),
                                       maxRadius=int(img_left.shape[0]))
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

            mask_left = self.hsv_mask(img_left, [[15, 50, 50], [35, 255, 255], [0, 0, 0], [0, 0, 0]], 21)

            mask_right = self.hsv_mask(img_right, [[170, 50, 50], [180, 255, 255], [0, 50, 50], [10, 255, 255]], 21)

            if ((mask_left > 0).sum() + eps)/mask_left.shape[0]/mask_left.shape[1] > state_thresh + eps:

                if ((mask_right > 0).sum() + eps)/mask_right.shape[0]/mask_right.shape[1] > state_thresh + eps:

                    return [True, True]

                else:
                    return [False, True]

            else:
                if ((mask_right > 0).sum() + eps) / mask_right.shape[0] / mask_right.shape[1] > state_thresh + eps:

                    return [True, False]

                else:
                    return [False, False]

        else:

            return None

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

    import os

    images_path = r'/home/hxzh/Wei_Work/Project/Dataset/2021-09-06/'
    dlq_type = 'ABB-SACE-Emax'
    Emax_ks = (31, 31)
    Emax_X1_ks = (11, 11)
    if dlq_type == 'ABB-SACE-Emax-X1':
        ks = Emax_X1_ks
    else:
        ks = Emax_ks
    images = os.listdir(images_path)
    DFind = Find(is_save=True, save_path='./crop/')
    for i, image_name in enumerate(images):
        image = cv2.imread(os.path.join(images_path, image_name))
        # image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
        crop_img = DFind.dlq_crop(image, ks, image_name, dlq_type)        # 31 11
        if crop_img is not None:
            temp_state = DFind.dlq_state(crop_img, dlq_type)
            image = cv2.putText(image, "({},{})".format(temp_state[0], temp_state[1]), (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
            image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('temp', image)
            print(image_name, 'state:', temp_state)
            cv2.waitKey(5000)
        else:
            print(image_name)

