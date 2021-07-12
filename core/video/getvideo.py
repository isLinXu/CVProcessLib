'''


'''

import cv2



def get_vedio():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('/home/linxu/Desktop/Work 快捷方式/sources/交通灯识别（重视红绿灯）/交通灯识别（重视红绿灯）/testwrite.avi', fourcc, 20.0, (1280, 720), True)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frame = cv2.resize(frame,(1280,720))
            cv2.imshow('frame', frame)
            out.write(frame)
            ret, frame = cap.read()

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def show_camera():
    cap = cv2.VideoCapture(0)
    return cap


if __name__ == "__main__":
    get_vedio()