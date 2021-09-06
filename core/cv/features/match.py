

import cv2
import numpy as np
if __name__ == '__main__':
    img1 = cv2.imread("/home/linxu/PycharmProjects/CVProcessLib/core/network/输电塔/3.jpeg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("/home/linxu/Desktop/无人机巡检项目/输电杆塔照片素材/输电杆塔照片素材/杆塔倒塌/1.JPG" , cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, (0,0), fx=0.1, fy=0.1)
    #detector
    # orb = cv2.ORB_create()
    orb = cv2.AKAZE_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2 , None)
    #for d in des1:
     #   print(d)
     #brute force matching
    bf =cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches , key = lambda x:x.distance)
    matching_result = cv2.drawMatches(img1 ,kp1, img2,kp2, matches, None )
    for m in matches:
       print(m.distance)

    cv2.imshow("img1" , img1)
    cv2.imshow("img2" , img2)
    cv2.imshow("matching result" , matching_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()