import xml.etree.ElementTree as ET
from PIL import Image
from object_detection.utils import label_map_util
from xml.etree.ElementTree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import dump
from xml.etree.ElementTree import Comment
from xml.etree.ElementTree import tostring
import os
from shutil import copyfile

filename="book.xml"
def CreateXml(name,imageName,path,width,height,xmin,ymin,xmax,ymax):
  book =ElementTree()
  purOrder =Element("annotation")
  book._setroot(purOrder)

  item = Element("folder")
  item.text=name
  purOrder.append(item)

  item = Element("filename")
  item.text = imageName
  purOrder.append(item)

  item = Element("path")
  item.text = path
  purOrder.append(item)

  item = Element("source")
  SubElement(item,"database").text="Unknown"
  # SubElement(item,"Description").text="My Country"
  purOrder.append(item)

  item = Element("size")
  SubElement(item, "width").text = str(width)
  SubElement(item,"height").text=str(height)
  SubElement(item, "depth").text = "3"
  purOrder.append(item)

  item = Element("segmented")
  item.text = "0"
  purOrder.append(item)

  item = Element("object")
  SubElement(item, "name").text = name
  SubElement(item, "pose").text = "Unspecified"
  SubElement(item, "truncated").text = "0"
  SubElement(item, "difficult").text = "0"


  nameE = Element('bndbox')
  SubElement(nameE, "xmin").text = str(xmin)
  SubElement(nameE, "ymin").text = str(ymin)
  SubElement( nameE, "xmax").text = str(xmax)
  SubElement( nameE, "ymax").text = str(ymax)
  item.insert(5, nameE)  # 方式三
  purOrder.append(item)
  # purOrder.append(nameE)
  indent(purOrder)
  return book
def indent(elem,level=0):
  i ="\n"+level*"  "
  # print elem;
  if len(elem):
    if not elem.text or not elem.text.strip():
      elem.text = i + "  "
    for e in elem:
      # print e
      indent(e,level+1)
    if not e.tail or not e.tail.strip():
      e.tail =i
  if level and (not elem.tail or not elem.tail.strip()):
    elem.tail =i
  return elem

def groupArray(bg_oriXml,save_i,save_c,ddic):

    dealt_xml=[]
    box_array=[]
    group_group_xml=[]
    for xmlfile in os.listdir(bg_oriXml):
        if (".xml" not in xmlfile) or (xmlfile in dealt_xml):
            continue
        # 读取 xml 文件
        # dom = xml.dom.minidom.parse(os.path.join(oriXml, xmlfile))
        tree = ET.parse(bg_oriXml + xmlfile)
        root = tree.getroot()
        # print(os.path.join(bg_oriXml, xmlfile))
        list_box=[]
        print(xmlfile)
        for element in root.findall('object'):
            dis_name=element.find("name").text
            # dis_name = ddic[element.find("name").text]
            if not os.path.exists(now_crop_image+dis_name):
                os.makedirs(now_crop_image+dis_name)
            box = (int(element.find("bndbox").find("xmin").text), int(element.find("bndbox").find("ymin").text),
                   int(element.find("bndbox").find("xmax").text), int(element.find("bndbox").find("ymax").text))
            save_c+=1
            # crop_save = Image.open(ori_image+root.findall('filename')[0].text)
            # magepath_crop = now_crop_image + element.find("name").text + "/" + str(save_c) + ".jpg"
            crop_save = Image.open(ori_image + xmlfile.split(".xml")[0]+".jpg")
            magepath_crop = now_crop_image + dis_name + "/" + str(save_c)+"_"+xmlfile.split(".xml")[0] + ".jpg"
            xx_save = crop_save.crop(box)
            xx_save.save(magepath_crop)
            list_box.append(box)
            # print(xmlfile+":"+str(box))
        # for element in root.findall('object'):
        #     box = (int(element.find("bndbox").find("xmin").text), int(element.find("bndbox").find("ymin").text),
        #            int(element.find("bndbox").find("xmax").text), int(element.find("bndbox").find("ymax").text))
        #     imBackground = Image.open(ori_image+root.findall('filename')[0].text)
        #     for box_dic in list_box:
        #         if box_dic != box:
        #             image = Image.open("/media/aihost/文档/image/bg.jpg")
        #             xx = image.crop(box_dic)
        #             imBackground.paste(xx, (box_dic[0],box_dic[1]))
        #             imBackground.save("/media/aihost/文档/image/bg11.jpg")
        #     print(xmlfile+":"+str(box))
        #     print(element.find("name").text)
        #     save_i += 1
        #     magepath=now_image+element.find("name").text+"/"+str(save_i) + ".jpg"
        #     xmlpath=now_xml+element.find("name").text+"/"+str(save_i) + ".xml"
        #     if not os.path.exists(now_xml+element.find("name").text):
        #         os.makedirs(now_xml+element.find("name").text)
        #     if not os.path.exists(now_image+element.find("name").text):
        #         os.makedirs(now_image+element.find("name").text)
        #     book = CreateXml(element.find("name").text, str(save_i) + ".jpg", magepath, str(1920),
        #                      str(1080), str(box[0]), str(box[1]), str(box[2]), str(box[3]))
        #     book.write(xmlpath, "utf-8")
        #     imBackground.save(magepath)  # 保存变化后的图片

    print("长度:",len(group_group_xml))
    print("大小:", group_group_xml)
    print("图片大小:", dealt_xml)
    return dealt_xml,box_array,group_group_xml

ii = 201912230000
ori_xml="/media/hxzh/CC06287006285DA8/image/实验室表记试验/喷绘版/xml/"
ori_image="/media/hxzh/CC06287006285DA8/image/实验室表记试验/喷绘版/image/"
# now_xml="/media/aihost/文档/python/lib/image/20191232/xml/"
# now_image="/media/aihost/文档/python/lib/image/20191232/image/"
now_crop_image="/media/hxzh/CC06287006285DA8/image/实验室表记试验/喷绘版/crop/"
PATH_TO_LABELS = os.path.join("/media/hxzh/CC06287006285DA8/svn", 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
# print("1111111111")
ddic={}
for k,c in category_index.items():
    ddic[c["name"]]=c["display_name"]
print(ddic)
groupArray(ori_xml,ii,ii,ddic)