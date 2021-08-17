#!/usr/bin/env python
import os, sys
import glob
import argparse
import cv2
import json
import numpy as np


LABEL_FILE = './labels.json'


def dump_label(node_list, filename):
    if os.path.isfile(LABEL_FILE):
        data = json.load(open(LABEL_FILE, 'r'))
    else:
        data = dict()
    if filename not in data:
        data[filename] = []
    data[filename].append(node_list)
    print(data)
    json.dump(data, open(LABEL_FILE, 'w'))


def labelling(imname):
    global drawing, ix, iy, cx, cy
    drawing = False
    win_name = "Image"
    node_list = []

    def showImage(src):
        global drawing, ix, iy, cx, cy
        img = src.copy()
        if drawing == True:
            cv2.line(img,(ix,iy),(cx,cy),(0,255,0),3)
        cv2.imshow(win_name, img)
        if cv2.waitKey(20) & 0xFF == 27:
            return True
        return False

    def near_to_start(x, y, tho=50):
        start_x, start_y = node_list[0]
        return np.sqrt(np.power(x-start_x, 2) + np.power(y-start_y, 2)) < tho

    def call_back(event, x, y, flags, param):
        global drawing, ix, iy, cx, cy
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing == False:
                drawing = True
                ix,iy = x,y
            else:
                if near_to_start(x,y):
                    cv2.line(im,(ix,iy),node_list[0],(0,255,0),3)
                    drawing = False
                else:
                    cv2.line(im,(ix,iy),(x,y),(0,255,0),3)
                    ix,iy = x,y
            node_list.append((x,y))
        elif event == cv2.EVENT_MOUSEMOVE:
            cx,cy = x,y

    im = cv2.imread(imname)
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, call_back)

    while(1):
        if showImage(im):
            break
    cv2.destroyAllWindows()
    # print(node_list)
    dump_label(node_list, os.path.basename(imname))
    print("label dumped to {}".format(LABEL_FILE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imagePath', type=str, help='Path of image files')
    parser.add_argument('annoFile', type=str, help='Annotation file')
    args = parser.parse_args()

    print("image path {}".format(args.imagePath))
    print("anno file {}".format(args.annoFile))

    LABEL_FILE = args.annoFile

    #labelling(args.filename)
    filelist = glob.glob(os.path.join(args.imagePath, '*.jpg'))
    for filename in filelist:
        labelling(filename)


if __name__ == '__main__':
    main()
