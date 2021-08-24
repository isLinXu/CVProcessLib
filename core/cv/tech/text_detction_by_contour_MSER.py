'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 04/11/2020
-- Description:	This code is for text detection using contours and 
                Maximally Stable Extremal Regions (MSER)feature detector. 
-- Status:      In progress
===============================================================================
'''

import glob
import sys
import os
import cv2
import csv
import imutils
import numpy as np
import pickle


import os.path


def canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def ii(xx, yy):
    global img, img_y, img_x
    if yy >= img_y or xx >= img_x:
        return 0
    pixel = img[yy][xx]
    return 0.30 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0]


def connected(contour):
    first = contour[0][0]
    last = contour[len(contour) - 1][0]
    return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1


def c(index):
    global contours
    return contours[index]


def count_children(index, h_, contour):
    if h_[index][2] < 0:
        return 0
    else:
        if keep(c(h_[index][2])):
            count = 1
        else:
            count = 0

        count += count_siblings(h_[index][2], h_, contour, True)
        return count


def is_child(index, h_):
    return get_parent(index, h_) > 0


def get_parent(index, h_):
    parent = h_[index][3]
    while not keep(c(parent)) and parent > 0:
        parent = h_[parent][3]

    return parent


def count_siblings(index, h_, contour, inc_children=False):
    if inc_children:
        count = count_children(index, h_, contour)
    else:
        count = 0

    p_ = h_[index][0]
    while p_ > 0:
        if keep(c(p_)):
            count += 1
        if inc_children:
            count += count_children(p_, h_, contour)
        p_ = h_[p_][0]

    n = h_[index][1]
    while n > 0:
        if keep(c(n)):
            count += 1
        if inc_children:
            count += count_children(n, h_, contour)
        n = h_[n][1]
    return count


def keep(contour):
    return keep_box(contour) and connected(contour)


def keep_box(contour):
    xx, yy, w_, h_ = cv2.boundingRect(contour)

    w_ *= 1.0
    h_ *= 1.0

    if w_ / h_ < 0.1 or w_ / h_ > 10:
        if DEBUG:
            print("\t Rejected because of shape: (" + str(xx) + "," + str(yy) + "," + str(w_) + "," + str(h_) + ")" +
                  str(w_ / h_))
        return False

    if ((w_ * h_) > ((img_x * img_y) / 5)) or ((w_ * h_) < 15):
        if DEBUG:
            print("\t Rejected because of size")
        return False

    return True


def include_box(index, h_, contour):
    if DEBUG:
        print(str(index) + ":")
        if is_child(index, h_):
            print("\tIs a child")
            print("\tparent " + str(get_parent(index, h_)) + " has " + str(
                count_children(get_parent(index, h_), h_, contour)) + " children")
            print("\thas " + str(count_children(index, h_, contour)) + " children")

    if is_child(index, h_) and count_children(get_parent(index, h_), h_, contour) <= 2:
        if DEBUG:
            print("\t skipping: is an interior to a letter")
        return False

    if count_children(index, h_, contour) > 2:
        if DEBUG:
            print("\t skipping, is a container of letters")
        return False

    if DEBUG:
        print("\t keeping")
    return True


DEBUG = 0
for imagePath in glob.glob(sys.argv[1]+"\\*.png"):
    filename = imagePath
    filename = filename.split("\\")[-1]
    row = cv2.imread(imagePath)
    orig_img = row

    img = row
    img_y = len(img)
    img_x = len(img[0])
    blue, green, red = cv2.split(img)

    blue_edges = cv2.Canny(blue, 200, 250)
    green_edges = cv2.Canny(green, 200, 250)
    red_edges = cv2.Canny(red, 200, 250)

    edges = blue_edges | green_edges | red_edges

    if DEBUG:
        print("Image is " + str(len(img)) + "x" + str(len(img[0])))

    contours, hierarchy = cv2.findContours(
        edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    hierarchy = hierarchy[0]

    if DEBUG:
        processed = edges.copy()
        rejected = edges.copy()

    keepers = []

    for index_, contour_ in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour_)

        if keep(contour_) and include_box(index_, hierarchy, contour_):
            keepers.append([contour_, [x, y, w, h]])
            if DEBUG:
                cv2.rectangle(processed, (x, y), (x + w, y + h),
                              (100, 100, 100), 1)
                cv2.putText(processed, str(index_), (x, y - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        else:
            if DEBUG:
                cv2.rectangle(rejected, (x, y), (x + w, y + h),
                              (100, 100, 100), 1)
                cv2.putText(rejected, str(index_), (x, y - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    new_image = edges.copy()
    new_image.fill(255)
    boxes = []

    for index_, (contour_, box) in enumerate(keepers):

        fg_int = 0.0
        for p in contour_:
            fg_int += ii(p[0][0], p[0][1])

        fg_int /= len(contour_)

        x_, y_, width, height = box
        bg_int = \
            [
                ii(x_ - 1, y_ - 1),
                ii(x_ - 1, y_),
                ii(x_, y_ - 1),

                ii(x_ + width + 1, y_ - 1),
                ii(x_ + width, y_ - 1),
                ii(x_ + width + 1, y_),

                ii(x_ - 1, y_ + height + 1),
                ii(x_ - 1, y_ + height),
                ii(x_, y_ + height + 1),

                ii(x_ + width + 1, y_ + height + 1),
                ii(x_ + width, y_ + height + 1),
                ii(x_ + width + 1, y_ + height)
            ]

        bg_int = np.median(bg_int)

        if fg_int >= bg_int:

            fg = 255
            bg = 0
        else:
            fg = 0
            bg = 255

        for x in range(x_, x_ + width):
            for y in range(y_, y_ + height):
                if y >= img_y or x >= img_x:
                    if DEBUG:
                        print("pixel out of bounds (%d,%d)" % (y, x))
                    continue
                if ii(x, y) > fg_int:
                    new_image[y][x] = bg
                else:
                    new_image[y][x] = fg

    mser = cv2.MSER_create()
    vis = img
    gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[0]
    regions = mser.detectRegions(new_image)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    for i, contour in enumerate(hulls):
        x, y, w, h = cv2.boundingRect(contour)

        if h > 45 and h < 60:
            print(filename)
            char = img[int(y):int(y+h)]
            cv2.imwrite(str(x)+"_"+str(y)+"_"+str(i)+"_"+filename, char)
