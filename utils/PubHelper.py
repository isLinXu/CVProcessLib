# -*- coding: utf-8 -*-
'''
@author: wuhang
@contact: whang@ikeeko.com
@time: 2018-06-04 下午18:12
@desc: 公用函数辅助
'''

import re, ujson, time, datetime, random, difflib, hashlib, base64, uuid
from datetime import timedelta

# ----------------------系统支持函数-------------------------
def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e+2] for e in range(0, 11, 2)])

# city = '福建省厦门市' 输出厦门
def GetIPLocation(city):
    shengpos = city.find('省')
    shipos = city.find('市')
    if shengpos > -1 and shipos > -1:
        city = city[shengpos+1:shipos]
    elif shengpos == -1 and shipos > -1:
        city = city[0:shipos]
    else:
        city = '北京'
    return city

# ----------------------字符串相关函数-------------------------
def str2json(data):
    """flask接到尖括号已经转义，字符串转json函数，处理尖括号"""
    data = str(data).replace('<', '&lt;')
    data = str(data).replace('>', '&gt;')
    data = ujson.loads(data)
    return data

def get_rank(strA, strB):
    """字符串相似度匹配"""
    seq = difflib.SequenceMatcher(None, str(strA), str(strB))
    ratio = seq.ratio()
    return float(ratio)

def check_pure_english(keyword):
    """判断是否纯英文"""
    return all(ord(c) < 128 for c in keyword)

def find_last(string, str):
    """改函数返回字符最后出现的位置"""
    last_position = -1
    while True:
        position = string.find(str, last_position + 1)
        if position == -1:
            return last_position
        last_position = position

def noEnglish_filter(text):
    """去英文过滤器"""
    return re.sub('[a-zA-Z]', '', text)

def Hans_filter(text):
    """保留中文过滤器"""
    tlong = len(text)
    wordList = []
    x = 0
    while x < tlong:
        X, Y = ['\u4e00', '\u9fa5']
        if X <= text[x] <= Y:
            wordList.append(text[x])
        else:
            wordList.append(" ")
        x = x + 1
    outStr = ''
    for wl in wordList:
        outStr = outStr + str(wl)
    return outStr

def GetSelectStr(text):
    """支持函数带（|）的递归筛选，以|平分概率，会达到某些概率不均"""
    if text:
        #先括号筛选
        pattern = r'(\(.*?\))'
        guid = re.findall(pattern, text, re.M)
        if (len(guid) > 0):
            for word in guid:
                word_content = word[(find_last(word, '(')):(len(word))]
                word_content_list = word_content.replace('(','').replace(')','')
                word_list = word_content_list.split('|')
                text_replace = str(random.sample(word_list, 1)[0])
                text = text.replace(word_content, text_replace, 1)
                text = GetSelectStr(text)
        #再整句筛选
        text_list = text.split('|')
        text = str(random.sample(text_list, 1)[0])
    else:
        text = ''
    return str(text)

def pattern_generate(pattern, depth=0):
    """外层括号嵌套函数"""
    pat_left = r'\((?:[^()]|'
    pat_right = r')*\)'
    while (depth):
        pattern = pat_left + pattern + pat_right
        depth -= 1
    return pattern

def GetDAGSelectStr(text, block_list = []):
    """从（|）筛选出多种可能的句子，平分概率选取"""
    if text:
        i = 0 #括号深度标识
        ii = []#括号分支标识 有10层做10层下标
        for p in range(10):
            ii.append(1)

        graph = {}
        graph['root'] = []
        def DAG_split(text, btext, i, ii):#i参数为层，ii为第几个位置
            i = i + 1
            text = text[1:len(text)-1]
            j = 0

            pattern_0 = r'\([^()]*\)'  # depth 0 pattern
            pat = pattern_generate(pattern_0, 10)
            prog = re.compile(pat)
            data = prog.findall(text)
            for word in data:
                text = text.replace(word, '#', 1)

            #再整句筛选
            text_list = text.split('|')
            n_text_list = text_list
            if i == 1: #句子样式标识
                a = list(range(1, len(text_list) + 1, 1))  # 代表从1到len(text_list)，间隔1(不包含len(text_list))
                n_text_list = [str(a[x]) + '&' + text_list[x] for x in range(len(text_list))]
            for word in n_text_list:
                #头结点root
                if i == 1:
                    if word.find('#') != -1:
                        graph['root'].append(word[:word.find('#')] + str(i) + str(ii[i]))
                    else:
                        #下标需计算
                        graph['root'].append(word + str(i) + str(ii[i]))
                #正式进入循环递归
                while (word.find('#') != -1): #取格式0#0
                    #前
                    f_text = word[:word.find('#')]

                    #词
                    m_text = data[j]
                    j = j + 1
                    m_text1 = m_text[1:len(m_text) - 1]#去括号
                    m_data = prog.findall(m_text1)
                    for m_word in m_data:
                        m_text1 = m_text1.replace(m_word, '#', 1)
                    m_text_list = m_text1.split('|')
                    m_new_text_list = []
                    p = ii[i +1]
                    for m_text_word in m_text_list:
                        if m_text_word.find('#') != -1:
                            pp = m_text_word.count('#')
                            m_text_word = m_text_word[:m_text_word.find('#')]
                            m_new_text_list.append(m_text_word + str(i +1) + str(p))
                            p = p + pp + 1
                        else:
                            m_new_text_list.append(m_text_word + str(i +1) + str(p))
                            p = p + 1
                    graph[f_text + str(i) + str(ii[i])] = m_new_text_list  # 前 接 词

                    #后
                    b_text_word = word[word.find('#') + 1:]
                    if b_text_word.find('#') > -1:  # 还有括号，为0#0#结构，则取到括号前做为后驱
                        b_text = b_text_word[:b_text_word.find('#')]
                    else:
                        b_text = b_text_word

                    DAG_split(m_text, b_text + str(i) + str(ii[i] + 1), i, ii) #因为进入下一个循环 所以加1

                    #更新参数值再进行循环
                    word = b_text_word
                    ii[i] = ii[i] + 1  # 括号分支标识
                #跳出循环
                if btext == '':
                    graph[word + str(i) + str(ii[i])] = []
                else:
                    graph[word + str(i) + str(ii[i])] = [btext]
                ii[i] = ii[i] + 1  # 括号分支标识
        DAG_split('('+text+')', '', i, ii)

        #屏蔽相关句式 block_list为一个list
        # for block in block_list[:]:
        #     graph['root'][int(block)] = 0
        # while 0 in graph['root']:
        #     graph['root'].remove(0)

        li = DAG_route(graph)
        data = str(random.sample(li, 1)[0])
        data_list = data.split('&')
        sdata = data_list[1]
        stype = data_list[0]
    else:
        sdata = ''
        stype = ''
    sdata_dict = {}
    sdata_dict['sdata'] = sdata #句子
    sdata_dict['stype'] = stype #样式标识
    return sdata_dict

# ----------------------DAG图相关函数-------------------------
def DAG_route(graph):
    """在DAG中遍历所有路径"""
    li = []
    sentense = ''
    def DAG_add(words, sentense):
        for word in words:
            if graph[word]:
                DAG_add(graph[word], sentense + word.rstrip('0123456789'))
            else:
                li.append(sentense + word.rstrip('0123456789'))
    if graph:
        DAG_add(graph['root'], sentense)
    return li

def topological_sort(graph):
    """在DAG中DFS中顶点的出栈顺序即逆拓扑序"""
    is_visit = dict((node, False) for node in graph)
    li = []
    def dfs(graph, start_node):
        for end_node in graph[start_node]:
            if not is_visit[end_node]:
                is_visit[end_node] = True
                dfs(graph, end_node)
        li.append(start_node)
    for start_node in graph:
        if not is_visit[start_node]:
            is_visit[start_node] = True
            dfs(graph, start_node)
    li.reverse()
    return li

# ----------------------时间相关函数-------------------------
def get_timeStamp(strtime):
    """字符串转时间戳"""
    timeArray = time.strptime(strtime, "%Y-% m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    #毫秒级时间
    return timedelta(microseconds=int(round(time_dif * 1000000)))

def todaystarttimestamp(timestamp):
    # """获取今天开始时间戳，方便计算当日筛选"""
    # today = datetime.date.today()
    # today_timestamp = int(time.mktime(time.strptime(str(today), '%Y-%m-%d')))
    """通过时间戳计算当日时间戳"""
    today_timestamp = int(time.mktime(time.strptime(time.strftime('%Y-%m-%d', time.localtime(timestamp)), '%Y-%m-%d')))
    return today_timestamp

# ----------------------加密相关函数-------------------------
def md5(str):
    """md5加密"""
    m = hashlib.md5()
    m.update(str)
    return m.hexdigest()

def get_base64(filepath):
    """base64编码"""
    encodestr = base64.b64encode(filepath.read())
    return encodestr

def getuuid():
    """获取通用唯一识别码"""
    l_uuid = str(uuid.uuid1()).split('-')
    s_uuid = ''.join(l_uuid)
    return str(s_uuid)


# if __name__ == '__main__':
#     ctime = time.time()
#     print(todaystarttimestamp1(ctime))
