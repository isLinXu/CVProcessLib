import re
import requests
from urllib import error
from bs4 import BeautifulSoup
import os

num = 0
numPicture = 0
file = ''#自定义文件夹(只写文件夹名就在py文件同级目录)，如果写了D://img就在D盘
header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'
    }


def countPic(url):

    print('正在检测图片总数，请稍等.....')
    pic_num=''
    t = 0#pn变量，用于拼接Url
    Url = url + str(t)
    try:
        html = requests.get(Url,header,timeout=7)
    except BaseException:
        print('请求异常')
    else:
        result = html.text
        #bdFmtDispNum: "约1,050,000",
        # print(result)
        flist = re.findall('"bdFmtDispNum":"(.*?)",', result, re.S)  # 先利用正则表达式找到图片总数
        # print(type(pic_num)) #list
        # print(type(flist))
        for i in flist:
            # print(i)
            pic_num+=i
    # print(pic_num)
    return pic_num



def recommend(url):
    Re = []
    try:
        html = requests.get(url)
    except error.HTTPError as e:
        return
    else:
        html.encoding = 'utf-8'
        bsObj = BeautifulSoup(html.text, 'html.parser')#bs4解析网页
        div = bsObj.find('div', id='topRS')
        if div is not None:
            listA = div.findAll('a')
            for i in listA:
                if i is not None:
                    Re.append(i.get_text())
        return Re



def dowmloadPicture(html, keyword):
    global num
    # t =0
    pic_url = re.findall('"thumbURL":"(.*?)"', html, re.S)  # 先利用正则表达式找到图片url
    print('找到关键词:' + keyword + '的图片，即将开始下载图片...')
    for each in pic_url:
        print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
        try:
            if each is not None:
                pic = requests.get(each, timeout=7)
            else:
                continue
        except BaseException:
            print('错误，当前图片无法下载')
            continue
        else:
            string = file + '/' + keyword + '_' + str(num) + '.jpg'
            fp = open(string, 'wb')
            fp.write(pic.content)
            fp.close()
            num += 1
        if num >= numPicture:
            return



if __name__ == '__main__':  # 主函数入口
    word = input("请输入搜索关键词(可以是人名，地名等): ")
    # 蛇的百度搜索图\\蛇的品种大全及图片_
    # url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='
    url='https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord='+word+'&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&word='+word+'&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&cg=girl&rn=30&gsm=&1575687587077=&pn='
    tot=countPic(url)
    Recommend = recommend(url)  # 记录相关推荐
    print('经过检测%s类图片共有%s张' % (word, tot))
    numPicture = int(input('请输入想要下载的图片数量 '))
    file = input('请建立一个存储图片的文件夹，输入文件夹名称即可')
    y = os.path.exists(file)
    if y == 1:
        print('该文件已存在，请重新输入')
        file = input('请建立一个存储图片的文件夹，)输入文件夹名称即可')
        os.mkdir(file)
    else:
        os.mkdir(file)
    t = 0
    tmp = url
    while t < numPicture:
        try:
            url = tmp + str(t)
            result = requests.get(url,header,timeout=100,allow_redirects=False)
            print(url)
        except error.HTTPError as e:
            print('网络错误，请调整网络后重试')
            t = t + 30
        else:
            dowmloadPicture(result.text, word)
            t = t + 30

    print('当前搜索结束，感谢使用')
    print('猜你喜欢')
    for re in Recommend:
        print(re, end='  ')
