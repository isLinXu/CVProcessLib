import cv2
import numpy as np
import matplotlib.pyplot as plt

img =cv2.imread("/home/hxzh02/桌面/44.jpg")
mask = np.zeros(img.shape[:2], np.uint8)

bg_model = np.zeros((1,65), np.float64)
fg_model = np.zeros((1,65), np.float64)

rect = (50,50,600,600)
cv2.grabCut(img, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)


mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar(),plt.show()