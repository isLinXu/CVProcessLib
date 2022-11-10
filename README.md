

![](img/cvprocesslib.png)

# CV-Process-Lib

---

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/isLinXu/CVProcessLib) ![img](https://badgen.net/badge/icon/vison?icon=awesome&label)
![](https://badgen.net/github/stars/isLinXu/CVProcessLib)![](https://badgen.net/github/forks/isLinXu/CVProcessLib)![](https://badgen.net/github/prs/isLinXu/CVProcessLib)![](https://badgen.net/github/releases/isLinXu/CVProcessLib)![](https://badgen.net/github/license/isLinXu/CVProcessLib)

#### 一、介绍

本项目为个人基于Opecv、numpy等Python开发包所建立维护的图像处理库。

#### 二、处理库架构

##### 2.1 基本介绍

该项目主要以Python为主要开发语言进行编写。

按照金字塔分层设计的思想，分为以下三层：

底层设计为独立模块[core]，主要为一些通用性、功能性的核心组件

中层设计为业务封装[package]，主要根据项目业务需求调用不同的组件和模型进行封装使用；

顶层设计为独立项目[project]，主要与公司产品业务需求进行对接。

##### 2.2 目录树结构

```shell
cvprocess-lib
├── core
│   ├── cv
│   │   ├── color
│   │   ├── contour
│   │   ├── features
│   │   ├── imgtransform
│   │   ├── light
│   │   ├── line
│   │   ├── ROI
│   │   ├── subtractor
│   │   ├── threshold
│   │   ├── tools
│   │   └── video
│   ├── log
│   ├── math
│   └── network
├── images
├── initize.py
├── package
├── README.md
├── temptest
├── testfunc.py
├── tools
└── utils
    ├── globals.py
    └── __init__.py
```

