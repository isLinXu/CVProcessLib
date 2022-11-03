from collections import Counter
from PIL import ImageDraw
import math
from sklearn.cluster import DBSCAN, Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

import numpy as np
import io

import time
from PIL import Image


class ColorExtractor(object):

    def __init__(self, COLOR_LIMIT=10, DIST_THRESHOLD=50, SIZE_LIMIT=1000, CLUSTER_LIMIT=9):
        self.COLOR_LIMIT = COLOR_LIMIT  # Max number of colors to display
        self.DIST_THRESHOLD = DIST_THRESHOLD
        self.SIZE_LIMIT = SIZE_LIMIT
        self.CLUSTER_LIMIT = CLUSTER_LIMIT

    def get_pixel_freq(self, image: Image.Image) -> Counter:
        return Counter(image.getdata())

    def get_pixel_list(self, image: Image.Image) -> list:
        return image.getdata()

    def get_dominant_colors_clustering(self, color_data: np.ndarray, model='kmeans'):
        # Mini-k means recommended for faster processing
        global yhat
        if model == "DBSCAN":
            model = DBSCAN(eps=0.30, min_samples=self.CLUSTER_LIMIT)
            yhat = model.fit_predict(color_data)

        elif model == "BIRCH":
            model = Birch(threshold=0.01, n_clusters=self.CLUSTER_LIMIT)
            model.fit(color_data)
            yhat = model.predict(color_data)

        elif model == "gaussian":
            model = GaussianMixture(n_components=self.CLUSTER_LIMIT)
            model.fit(color_data)
            yhat = model.predict(color_data)

        elif model == "kmeans":
            model = MiniBatchKMeans(n_clusters=self.CLUSTER_LIMIT)
            model.fit(color_data)
            yhat = model.predict(color_data)

        return yhat

    def get_color_pallete(self, image: Image.Image, model='kmeans'):

        image = self.preprocess_image(image)

        color_image = Image.Image.getdata(image)
        color_data = np.array(color_image)


        yhat = self.get_dominant_colors_clustering(color_data, model=model)

        clusters = np.unique(yhat)

        color_pallete = []

        for cluster in clusters:
            idx = np.where(yhat == cluster)

            color = (
                round(np.average(color_data[idx, 0])),
                round(np.average(color_data[idx, 1])),
                round(np.average(color_data[idx, 2]))
            )

            color_pallete.append(color)

        return color_pallete

    # Color pallete in RGB format
    def visualize_color_pallete(self, color_pallete: list):
        new_img = Image.new(mode='RGB', size=(100 * len(color_pallete), 200), color='white')
        draw = ImageDraw.Draw(new_img)

        for idx, color in enumerate(color_pallete):
            draw.text((idx * 100 + 25, 50), f"{self.rgb_to_hex(color)}", (0, 0, 0))

            new_img.paste(color, (idx * 100, 100, idx * 100 + 100, 200))

        new_img.show()

        return new_img

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % rgb

    def hex_to_rgb(self, hex):
        hex = hex[1:]
        return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))

    def preprocess_image(self, image: Image.Image):

        if type(image) == io.BufferedRandom:
            buf = io.BytesIO()
            image.write(buf)
            buf.seek(0)
            image = buf

        try:
            new_image = Image.open(image)
        except:
            return Exception("ERROR: not openable image")

        max_side = max(new_image.width, new_image.height)

        if max_side > self.SIZE_LIMIT:
            scale_factor = max_side / self.SIZE_LIMIT
            new_image = new_image.transform(
                (int(new_image.width / scale_factor), int(new_image.height // scale_factor)), Image.EXTENT,
                data=[0, 0, new_image.width, new_image.height])

        # new_image.show()
        return new_image

    # Old method
    # def get_dominant_colors(self, color_dict: Counter):
    #     for c1 in list(color_dict.keys()):
    #         for c2 in list(color_dict.keys()):
    #             dist = math.dist(c1, c2)
    #             if dist < self.DIST_THRESHOLD and dist != 0:
    #                 weighted_color = self.get_weighted_average(c1, color_dict[c1], c2, color_dict[c2])
    #                 color_dict[weighted_color] += color_dict[c1] + color_dict[c2]
    #                 del color_dict[c1]
    #                 del color_dict[c2]
    #                 break
    #     return color_dict


if __name__ == "__main__":
    img_path = '/home/linxu/Desktop/voc128/JPEGImages/000000000009.jpg'
    img = Image.open(img_path)
    # img = Image.open(input("Enter path/name for image: "))

    ce = ColorExtractor()

    start = time.time()
    color_p = ce.get_color_pallete(img)
    print("time elapsed: ", time.time() - start)
    print("color pallete:", color_p)
    ce.visualize_color_pallete(color_p)

    start = time.time()
    color_p = ce.get_color_pallete(img, model='gaussian')
    print("time elapsed: ", time.time() - start)
    print("color pallete:", color_p)
    ce.visualize_color_pallete(color_p)
