import cv2
import numpy as np

global img, point
global inpaintMask
#手动祛痘
def manual_acne(event, x, y, flags, param):
    global img, point
    img2 = img.copy()
    height, width, n = img.shape
    inpaintMask = np.zeros((height, width), dtype=np.uint8)
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        cv2.circle(img2, point, 15, (0, 255, 0), -1)
        cv2.circle(inpaintMask, point, 15, 255, -1)
        cv2.imshow("image", img2)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img2, point, 15, (0, 255, 0), -1)
        cv2.circle(inpaintMask, point, 15, 255, -1)
        cv2.imshow("inpaintMask", inpaintMask)
        cv2.imshow("image", img2)
        cv2.imshow("image0", img)
        result = cv2.inpaint(img, inpaintMask, 100, cv2.INPAINT_TELEA)
        cv2.setMouseCallback("image", manual_acne)
        cv2.imshow("result", result)


if __name__ == "__main__":
    global img
    img = cv2.imread("/home/linxu/Desktop/face.png")
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", manual_acne)
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
