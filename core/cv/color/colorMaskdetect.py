import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np


# Define some helper functions that will help us to plot the images and the contours
def show(image):
    plt.figure(figsize=(10, 12))
    plt.imshow(image, interpolation='nearest')


def show_hsv(hsv):
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    show(rgb)


def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')


def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)

    return img


def find_biggest_contour(image):
    image = image.copy()
    im2,contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in    contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

    return biggest_contour, mask


def circle_countour(image, countour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(countour)
    cv2.ellipse(image_with_ellipse, ellipse, (0,255,0), 2)

    return image_with_ellipse


# First load the image and examine the properties of that image, such as the color spectrum and the dimensions.
image = cv2.imread('images/parking_cars.jpg')

# Since the order of the image stored in the memory is Blue Green Red (BGR), we need to convert it into RGB.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# We will be scaling down the image dimensions.
max_dimension = max(image.shape)
scale = 700 / max_dimension
image = cv2.resize(image, None, fx=scale,fy=scale)

# Gaussian filters are very popular in the research field and are used for various operations,
# one of which is the blurring effect that reduces the noise and balances the image.
image_blur = cv2.GaussianBlur(image, (7, 7), 0)

# Then we will convert the RGB-based image into an HSV color spectrum, which will help us
# to extract other characteristics of the image using color intensity, brightness, and shades.
image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

red = np.uint8([[[255,28, 28]]])
hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

# We need to create a mask that can detect the specific color RED, in our case.
min_red = np.array([0, 100, 100])
max_red = np.array([10, 255, 255])
mask1 = cv2.inRange(image_blur_hsv, min_red, max_red) # Filter by color
min_red = np.array([170, 100, 80])
max_red = np.array([180, 255, 255])
mask2 = cv2.inRange(image_blur_hsv, min_red, max_red) # Filter by brightness
mask = mask1 + mask2 # Concatenate both the mask for better feature extraction

# Once we are able to create our mask successfully, we need to perform some morphological
# operations, which are basic image processing operations used for the analysis and
# processing of geometrical structures.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))

# MORPH_CLOSE: Helpful to close small pieces inside the foreground objects or small black points on the object.
mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# MORPH_OPEN: The opening operation erosion followed by dilation is used to remove noise.
mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

# It's time to use the mask that we created to extract the object from the image.
big_contour, red_mask = find_biggest_contour(mask_clean) # Extract biggest bounding box
overlay = overlay_mask(red_mask, image) # Apply mask
circled = circle_countour(overlay, big_contour) # Draw bounding box

# Voila! So, we successfully extracted the image and also drew the bounding box around the object
show(circled)