#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import json
import os
import os.path as osp
import PIL.Image
import yaml
from labelme import utils
import cv2
from skimage import img_as_ubyte


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    list_path = os.listdir(json_file)
    print('json_file:', json_file)
    for i in range(0, len(list_path)):
        path = os.path.join(json_file, list_path[i])
        if os.path.isfile(path):

            data = json.load(open(path))
            img = utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            lbl_viz = utils.draw_label(lbl, img, captions)

            out_dir = osp.basename(path).replace('.', '_')
            save_file_name = out_dir

            # ------------------------保存从json中解析出来的图像、label、图像+label-------------------
            if not osp.exists(json_file + '/' + 'labelme_json'):
                os.mkdir(json_file + '/' + 'labelme_json')
            labelme_json = json_file + '/' + 'labelme_json'

            out_dir1 = labelme_json + '/' + save_file_name
            if not osp.exists(out_dir1):
                os.mkdir(out_dir1)

            PIL.Image.fromarray(img).save(out_dir1 + '/' + save_file_name + '_img.png')
            PIL.Image.fromarray(lbl).save(out_dir1 + '/' + save_file_name + '_label.png')
            PIL.Image.fromarray(lbl_viz).save(out_dir1 + '/' + save_file_name + '_label_viz.png')

            # ---------------------------------保存label的mask（0 1 2 3）----------------------------
            if not osp.exists(json_file + '/' + 'mask_png'):
                os.mkdir(json_file + '/' + 'mask_png')
            mask_save2png_path = json_file + '/' + 'mask_png'

            mask_dst = img_as_ubyte(lbl)  # mask_pic
            print('pic2_deep:', mask_dst.dtype)
            cv2.imwrite(mask_save2png_path + '/' + save_file_name + '_label.png', mask_dst * 50)

            with open(osp.join(out_dir1, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')

            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir1, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            print('Saved to: %s' % out_dir1)


if __name__ == '__main__':
    main()
