import cv2
import numpy as np

def hog_svm(im, show=False):
    # im = cv2.imread(img_path)
    org_h, org_w = im.shape[:2]
    size = (512, int(512 * org_h / org_w))
    im = cv2.resize(im, size)
    #hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #hog = cv2.HOGDescriptor((32,64), (8,8), (4,4), (4,4), 9)
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
    #hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
    hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    locations, r = hog.detectMultiScale(im, winStride=(8, 8), padding=(32, 32), scale=1.05, hitThreshold=0, finalThreshold=1)
    for (x, y, w, h) in locations:
        cv2.rectangle(im, (x, y),(x+w, y+h),(255,255,0), 3)
    if show:
        cv2.imshow("detect image",im)
        # cv2.waitKey(0)
    return im


def hog_svm(im, show=False):
    # im = cv2.imread(img_path)
    org_h, org_w = im.shape[:2]
    size = (512, int(512 * org_h / org_w))
    im = cv2.resize(im, size)
    # face = cv2.face_BasicFaceRecognizer()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # hog = cv2.HOGDescriptor((32,64), (8,8), (4,4), (4,4), 9)
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
    #hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
    # hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
    # face.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    locations, r = hog.detectMultiScale(im, winStride=(8, 8), padding=(32, 32), scale=1.05, hitThreshold=0, finalThreshold=1)
    for (x, y, w, h) in locations:
        cv2.rectangle(im, (x, y),(x+w, y+h),(255,255,0), 3)
    if show:
        cv2.imshow("detect image",im)
        # cv2.waitKey(0)
    return im


if __name__ == '__main__':
    url = 'rtsp://192.168.42.1/live'

    cap = cv2.VideoCapture(url)
    while (cap.isOpened()):
        try:
            ret, frame = cap.read()
            # cv2.imshow('frame', frame)

            # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #
            # blur = cv2.GaussianBlur(grey, (5, 5), 0)
            # ret, th = cv2.threshold(blur, 20, 255, cv2.THRESqH_BINARY)
            # dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
            # coutour, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #
            # cv2.drawContours(frame, coutour, -1, (0, 255, 0), 2)

            cv2.imshow('src', frame)
            dst_im = hog_svm(frame, True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except:
            print('not frame')
            cap = cv2.VideoCapture(url)
            ret, frame = cap.read()
            cv2.imshow('src', frame)
            dst_im = hog_svm(frame, True)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()