

import socket

import sys

# 通过socket判断部署环境
localhost_name = socket.gethostname()

if localhost_name.__contains__('ubuntu'):
    # 根据启动命令 gunicorn -c gunicorn.py run:app pro 区分开发或生产环境
    ENV = sys.argv[-1]
    # 按照服务器规定
    # dev为开发环境
    # pro为生产环境
    ENV_DICT = {"dev3": "http://dev.r.open.keeko.ai", "pro3": "http://r.open1.keeko.ai",
                "pro4": "http://r.open2.keeko.ai"}
    domain = ENV_DICT.get(ENV, '')

    # 为避免地址被网络查看，增加代号
    ENV_CODE = {"dev3": "dev", "pro3": "a", "pro4": "b"}
    ENV_PATH = '/' + ENV_CODE.get(ENV, '')
else:
    domain = 'http://localhost'
    ENV_PATH = ''
