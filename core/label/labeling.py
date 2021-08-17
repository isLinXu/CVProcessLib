import cv2 as cv
import os
import numpy as np
import json
import argparse

global img, point1, point2, points
drawing = False


def OnMouse(event, x, y, flag, params):
    global img, point1, point2, drawing, points
    img2 = img.copy()

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        point1 = (x, y)
        cv.imshow('image', img2)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            point2 = (x, y)
            cv.rectangle(img2, point1, point2, (0, 0, 255), 1)
            cv.imshow('image', img2)

    elif event == cv.EVENT_LBUTTONUP:
        cv.rectangle(img2, point1, point2, (0, 0, 255), 1)
        cv.imshow('image', img2)
        drawing = False
        points = (point1, point2)


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('p', type=str, help='path of the image folder')
    args = parser.parse_args()
    return vars(args)


def main(path):
    global img, points
    points = ((0, 0), (0, 0))
    imgtypedic = {'.png', '.jpg','.jpeg'}
    # args = getArgs()
    # path = args['p']

    i = 0
    while (1):
        filenames = os.listdir(path)
        # print(i)
        if i <= 0:
            i = 0
        if i >= len(filenames):
            i = len(filenames) - 2

        if os.path.splitext(filenames[i])[1] not in imgtypedic:
            i += 1
            continue
        print(filenames[i])

        try:
            # if os.path.getsize(os.path.join(path, filenames[i]) + '.txt') != 0:
            with open(os.path.join(path, os.path.splitext(filenames[i])[0]) + '.txt') as f:
                js = f.read()
                data = json.loads(js)
                points = (tuple(data[0]), tuple(data[1]))
                # print(points)
        except FileNotFoundError:
            points = ((0, 0), (0, 0))

        img = cv.imread(os.path.join(path, filenames[i]))

        cv.rectangle(img, points[0], points[1], (0, 0, 255), 1)

        #     # cv.imshow('image', img)
        #     # img=cv.resize(img,None,fx=0.4,fy=0.4)

        cv.namedWindow('image')
        cv.setMouseCallback('image', OnMouse)
        cv.imshow('image', img)
        k = cv.waitKey(0) & 0xff

        if k == 27:
            with open(os.path.join(path, os.path.splitext(filenames[i])[0]) + '.txt', mode='w') as f:
                f.write(json.dumps(points))
            return 0
        elif k == ord('n'):
            with open(os.path.join(path, os.path.splitext(filenames[i])[0]) + '.txt', mode='w') as f:
                f.write(json.dumps(points))
            i += 1

        elif k == ord('m'):
            with open(os.path.join(path, os.path.splitext(filenames[i])[0]) + '.txt', mode='w') as f:
                f.write(json.dumps(points))
            i -= 2

        elif k == ord('r'):
            points = ((0, 0), (0, 0))
            with open(os.path.join(path, os.path.splitext(filenames[i])[0]) + '.txt', mode='w') as f:
                f.write(json.dumps(points))

    cv.destroyAllWindows()

if __name__ == '__main__':
    path = '/home/hxzh02/PycharmProjects/cvprocess-lib/images/表计'
    main(path)