# -*- coding: UTF-8 -*-

import os
import imageio


def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        if image_name.endswith('.png'):
            print(image_name)
            frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)

    return


def main():
    path = r'/home/linxu/Desktop/Img2gif/Example frame/'
    image_list = [path + img for img in os.listdir(path)]
    gif_name = 'created_gif5.gif'
    create_gif(image_list, gif_name)


if __name__ == "__main__":
    main()