from xml.dom import minidom
import cv2
from django.conf import settings
import os
import numpy

url = os.path.join(settings.BASE_DIR, 'Files')


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        # 十六进制格式颜色
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        # 返回对应颜色
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        # 16进制的颜色格式转为RGB格式
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def fileread(fiile, immage):
    file = minidom.parse(fiile)
    fname = file.getElementsByTagName('filename')
    sname = fname[0].firstChild.data
    objkt = file.getElementsByTagName('name')
    objec = [i.firstChild.data for i in objkt]
    xmin = file.getElementsByTagName('xmin')
    xmax = file.getElementsByTagName('xmax')
    ymin = file.getElementsByTagName('ymin')
    ymax = file.getElementsByTagName('ymax')
    coor = {objec[m]: list(map(int, [i.firstChild.data, j.firstChild.data, k.firstChild.data, l.firstChild.data])) for
            i, j, k, l, m in zip(xmin, ymin, xmax, ymax, range(len(objec)))}

    for i, j in zip(range(len(objec)), coor.values()):
        imagedata.objects.create(filename=sname, object_name=objec[i], xmin=j[0], ymin=j[1], xmax=j[2], ymax=j[3])

    image = cv2.imdecode(numpy.fromstring(immage, numpy.uint8), cv2.IMREAD_UNCHANGED)
    # color = (0,0,255)
    # color = colors(c,True)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    filename = "default.jpg"

    for (i, j) in zip(coor.values(), range(len(objec))):
        start_point = (i[0], i[1])
        end_point = (i[2], i[3])
        org = (i[0] - 20, i[1] - 20)
        color = colors(j, True)
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        image = cv2.putText(image, objec[j], org, font, fontscale, color, thickness, cv2.LINE_AA)
    if os.path.isfile(os.path.join(url, 'default.jpg')):
        os.remove(os.path.join(url, 'default.jpg'))
    os.chdir(url)
    cv2.imwrite(filename, image)