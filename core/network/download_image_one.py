# 抓取指定网页所有图片保存到本地
import requests
import os
from urllib.parse import urlparse
from lxml import etree as et
import re
# 请求头
class DownloadImage(object):
    headers = {
    # 用户代理
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'
    }
    _downloadDir = './img/'

    def __init__(self,url,download_path=None,filter=[]):
        self.url = url
        self.initUrl()
        self.filter =filter

        # 定义图片下载图径
        if download_path:
            self.downloadPath=self._downloadDir + download_path
        else:
            self.downloadPath=self._downloadDir + self.urlParse.netloc
        self.makeDir()
        self.getImages()
        print(self.Imageurls)

    #通用图片路径方法格式化
    def initUrl(self):
        self.urlParse=urlparse(self.url)

    def getImages(self):
        response = requests.get(self.url, headers=self.headers)
        if response.status_code == 200:
            html = et.HTML(response.text)
            images = html.xpath('//img/@src')
            print(self.filter)
            match = '|'.join(self.filter)
            self.Imageurls = []
            for value in images:
                if not re.search(match,value):
                    self.Imageurls.append(value)
        else:
            return None

    #格式化图片URL
    def formatImageUrls(self,url):
        imgParase = urlparse(url)
        if not imgParase.netloc:
            imgpath = "%s://%s%s" %(self.urlParse.scheme,self.urlParse.netloc,imgParase.path)
        else:
            imgpath = url
        return imgpath
    # 保存图片
    def downloadImage(self,url):
        print("download :" + url)
        arr = url.split('/')
        file_name = self.downloadPath +'/' + arr[-1]
        # file_name = self.downloadPath +'/' + arr[-2] +'/' + arr[-1]
        response = requests.get(url, headers=self.headers)
        with open(file_name, 'wb') as fp:
            for data in response.iter_content(128):
                fp.write(data)
    def makeDir(self):
        if not os.path.exists(self.downloadPath):
            os.makedirs(self.downloadPath)

    def run(self):
        for img in self.Imageurls:
            self.downloadImage(self.formatImageUrls(img))

if __name__ == '__main__':

    # url = input("input Url:")
    url='https://www.51tietu.net'
    obj=DownloadImage(url,None,['png','gif'])
    #过滤格式
    obj.run()