import xml.etree.ElementTree as ET
import os
import json

voc_clses = ['1_0_0_1_1_0', '0_0_0_1_1_0', '1_0_0_1_2_0', '1_0_0_1_3_0', '0_0_0_1_3_0', '1_0_0_1_4_0', '0_0_0_1_4_0',
             '0_0_0_3_4_0', '0_0_0_1_5_0', '0_0_0_1_12_0', '0_0_0_1_6_0', '1_0_0_1_6_0', '0_0_0_3_6_0', '1_0_0_1_10_0',
             '0_0_0_1_7_0', '0_0_0_1_8_0', '1_0_0_1_11_0', '0_0_0_1_9_0', '1_0_0_1_20_0', '0_0_0_1_21_0',
             '1_0_0_1_22_0', '0_0_0_1_22_0', '1_0_0_2_29_0', '0_0_0_10_0_0', '0_0_0_11_0_0', '0_0_0_12_0_0',
             '0_0_0_20_0_0', '0_0_0_28_0_0', '0_0_0_50_0_0', '1_0_6_21_30_0', '1_0_8_21_31_0', '1_0_6_21_32_0',
             '1_0_6_21_33_0', '1_0_6_21_34_0', '1_0_6_21_35_0', '1_0_6_21_36_0', '1_0_3_22_40_0', '1_0_3_22_42_0',
             '1_0_4_22_43_0', '0_0_0_1_13_0', '1_0_3_22_41_0', '0_0_0_30_2_0', '0_0_0_30_3_0', '0_0_0_30_4_0',
             '0_0_0_30_5_0', '0_0_0_30_6_0', '0_0_0_30_7_0', '0_0_0_30_8_0', '0_0_0_40_1_0', '0_0_0_40_2_0',
             '0_0_0_40_3_0',
             '0_0_0_40_4_0', '0_0_0_40_5_0', '0_0_0_40_6_0', '0_0_0_40_8_0', '0_0_0_40_9_0', '0_0_0_40_10_0',
             '0_0_0_40_11_0', '0_0_0_40_12_0']

categories = []
for iind, cat in enumerate(voc_clses):
    cate = {}
    cate['supercategory'] = cat
    cate['name'] = cat
    cate['id'] = iind + 1
    categories.append(cate)


def getimages(xmlname, id):
    sig_xml_box = []
    tree = ET.parse(xmlname)
    root = tree.getroot()
    images = {}
    for i in root:  # 遍历一级节点
        if i.tag == 'filename':
            file_name = i.text  # 0001.jpg
            # print('image name: ', file_name)
            images['file_name'] = file_name
        if i.tag == 'size':
            for j in i:
                if j.tag == 'width':
                    width = j.text
                    images['width'] = int(width)
                if j.tag == 'height':
                    height = j.text
                    images['height'] = int(height)
        if i.tag == 'object':
            for j in i:
                if j.tag == 'name':
                    cls_name = j.text
                cat_id = voc_clses.index(cls_name) + 1
                if j.tag == 'bndbox':
                    bbox = []
                    xmin = 0
                    ymin = 0
                    xmax = 0
                    ymax = 0
                    for r in j:
                        if r.tag == 'xmin':
                            xmin = eval(r.text + ".0")
                        if r.tag == 'ymin':
                            ymin = eval(r.text + ".0")
                        if r.tag == 'xmax':
                            xmax = eval(r.text + ".0")
                        if r.tag == 'ymax':
                            ymax = eval(r.text + ".0")
                    bbox.append(xmin)
                    bbox.append(ymin)
                    bbox.append(xmax - xmin)
                    bbox.append(ymax - ymin)
                    bbox.append(id)  # 保存当前box对应的image_id
                    bbox.append(cat_id)
                    # anno area
                    bbox.append((xmax - xmin) * (ymax - ymin) - 10.0)  # bbox的ares
                    # coco中的ares数值是 < w*h 的, 因为它其实是按segmentation的面积算的,所以我-10.0一下...
                    sig_xml_box.append(bbox)
                    # print('bbox', xmin, ymin, xmax - xmin, ymax - ymin, 'id', id, 'cls_id', cat_id)
    images['id'] = id
    # print ('sig_img_box', sig_xml_box)
    return images, sig_xml_box


def txt2list(txtfile):
    f = open(txtfile)
    l = []
    for line in f:
        l.append(line[:-1])
    return l


# segmentation
def get_seg(points):
    min_x = points[0]
    min_y = points[1]
    max_x = points[2]
    max_y = points[3]
    h = max_y - min_y
    w = max_x - min_x
    a = []
    a.append([min_x, min_y, min_x, min_y + 0.5 * h, min_x, max_y, min_x + 0.5 * w, max_y, max_x, max_y, max_x,
              max_y - 0.5 * h, max_x, min_y, max_x - 0.5 * w, min_y])
    return a


# 计算面积
def get_area(points):
    min_x = points[0]
    min_y = points[1]
    max_x = points[2]
    max_y = points[3]
    return (max_x - min_x + 1) * (max_y - min_y + 1)


# voc2007xmls = 'anns'
voc2007xmls = '/media/hxzh/CC06287006285DA8/image/训练图片/20200420实验室/xml/'
# test_txt = 'voc2007/test.txt'
# test_txt = '/data2/chenjia/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
# xml_names = txt2list(test_txt)
xmls = []
bboxes = []
ann_js = {}
for xml_name in os.listdir(voc2007xmls):
    xmls.append(os.path.join(voc2007xmls, xml_name))
json_name = 'data/shiyanshi.json'
images = []
for i_index, xml_file in enumerate(xmls):
    image, sig_xml_bbox = getimages(xml_file, i_index + 1)
    images.append(image)
    bboxes.extend(sig_xml_bbox)
ann_js['images'] = images
ann_js['categories'] = categories
annotations = []
for box_ind, box in enumerate(bboxes):
    anno = {}
    anno['image_id'] = box[-3]
    anno['category_id'] = box[-2]
    anno['bbox'] = box[:-3]
    anno['id'] = box_ind + 1
    anno['area'] = get_area(box[:-3])
    anno['segmentation'] = get_seg(box[:-3])
    anno['iscrowd'] = 0
    annotations.append(anno)
ann_js['annotations'] = annotations
json.dump(ann_js, open(json_name, 'w'), indent=4)  # indent=4 更加美观显示
