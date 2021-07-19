# CV-Process-Lib

#### 一、介绍

本项目为海峡智汇研发中心算法部门图像算法组为公司机器人视觉处理需要专门建立的图像处理库。

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
│   │   │   ├── colorcut.py
│   │   │   ├── colorDection.py
│   │   │   ├── colorMatching.py
│   │   │   ├── ColorRecognition.py
│   │   │   └── selectColor.py
│   │   ├── contour
│   │   │   ├── prewitt.py
│   │   │   └── selectCanny.py
│   │   ├── features
│   │   │   ├── features.py
│   │   │   └── globals.py
│   │   ├── imgtransform
│   │   │   ├── RotationCorrection.py
│   │   │   ├── rotation.py
│   │   │   └── scaleDown.py
│   │   ├── light
│   │   │   └── light_adapt.py
│   │   ├── line
│   │   │   └── LineDetection.py
│   │   ├── ROI
│   │   │   ├── DP_ROI.py
│   │   │   └── selectROI.py
│   │   ├── subtractor
│   │   │   ├── BackgroundSubtractor.py
│   │   │   ├── grabcut.py
│   │   │   ├── panelAbstractCut.py
│   │   │   └── test.py
│   │   ├── threshold
│   │   │   ├── thresholdSegmentations.py
│   │   │   └── thresholdSplit.py
│   │   ├── tools
│   │   │   ├── enhancement.py
│   │   │   ├── image
│   │   │   │   └── test.png
│   │   │   ├── image_color.py
│   │   │   ├── image_enhancement.py
│   │   │   ├── image_filtering.py
│   │   │   ├── image_outline.py
│   │   │   ├── image_transformation.py
│   │   │   ├── main.py
│   │   │   └── utils.py
│   │   └── video
│   │       ├── getvideo.py
│   │       └── videoprocess.py
│   ├── log
│   │   └── err_log
│   │       └── except_err.py
│   ├── math
│   │   └── hx_math.py
│   └── network
│       └── Crawler_downloads_Baidu_pictures.py
├── images
│   ├── 表计
│   ├── 呼吸器
│   ├── 数显表
│   └── 状态指示器
├── initize.py
├── LICENSE
├── package
│   ├── Respirator_Recognition.py
│   └── StatusIndicator_Recognition.py
├── README.md
├── temptest
│   └── result.py
├── testfunc.py
├── tools
│   ├── cut_label_image.py
│   └── XML_tran_coco.py
└── utils
    ├── globals.py
    └── __init__.py
```



软件架构说明


#### 三、依赖项

1.  xxxx
2.  xxxx
3.  xxxx

#### 四、使用说明

1.  xxxx
2.  xxxx
3.  xxxx

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request

6.  https://gitee.com/gitee-stars/)
