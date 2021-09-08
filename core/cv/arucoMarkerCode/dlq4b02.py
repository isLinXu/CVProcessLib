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

    def dlq_location(self, img, ks=None, save_name=None, dlq_type=None):

        if ks is not None:
            img_ = cv2.GaussianBlur(img, ks, 0)
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
        # start_func_time = time.time()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        # print('func_time:', time.time() - start_func_time)
        if ids is not None and len(corners) != 0:

            aruco.drawDetectedMarkers(img, corners, ids)
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

            # lower = point1[1]
            # upper = point1[1] - (point2[1] - point1[1]) * 1.1
            # left = point4[0] - (point1[0] - point4[0]) * 1.5
            # right = point1[0] + (point1[0] - point4[0]) * 1.5

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


if __name__ == '__main__':

    import os

    images_path = r'/home/hxzh/Wei_Work/Project/Dataset/0908/'
    dlq_type = 'ABB-SACE-Emax-X1'
    images = os.listdir(images_path)
    DFind = Find(is_save=True, save_path='./crop/')
    for i, image_name in enumerate(images):
        image = cv2.imread(os.path.join(images_path, image_name))
        # image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
        crop_img = DFind.dlq_location(image, (31, 31), image_name, dlq_type)
        if crop_img is not None:
            print(i+1)
            cv2.imshow('temp', crop_img)
            cv2.waitKey(500)
        else:
            print(image_name)

