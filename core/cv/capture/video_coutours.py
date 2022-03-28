import cv2
import numpy as np

def coutours_capture():
    cap = cv2.VideoCapture(-1)

    while (True):
        ret, frame = cap.read()
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        d = cv2.absdiff(frame1, frame2)

        # gray = cv2.cvtColor(frame, cv2.COLOR_B)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        #
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
        coutour, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        cv2.drawContours(d, coutour, -1, (0, 255, 0), 2)

        cv2.imshow('frame', d)
        cv2.imshow('src',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    coutours_capture()