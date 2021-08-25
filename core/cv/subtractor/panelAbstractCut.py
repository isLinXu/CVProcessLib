
'''
Extract panel :kmeans聚类
函数的格式为：kmeans(data, K, bestLabels, criteria, attempts, flags)
（1）data: 分类数据，最好是np.float32的数据，每个特征放一列。之所以是np.float32原因是这种数据类型运算速度快，同样的数据下如果是uint型数据将会慢死你。
(2) K: 分类数，opencv2的kmeans分类是需要已知分类数的。
(3) bestLabels：预设的分类标签：没有的话 None
(4) criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type,max_iter,epsilon）
其中，type又有两种选择：
—–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
—- cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
—-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
（5）attempts：重复试验kmeans算法次数，将会返回最好的一次结果
（6）flags：初始类中心选择，两种方法
cv2.KMEANS_PP_CENTERS ; cv2.KMEANS_RANDOM_CENTERS
原文链接：https://blog.csdn.net/Dawn__Z/article/details/82115160
'''


import cv2
import numpy as np
import math

def panelAbstract(srcImage):
    '''
    图片前景与背景的分割
    :param srcImage:
    :return:
    '''
    #   read pic shape
    imgHeight,imgWidth = srcImage.shape[:2]
    imgHeight = int(imgHeight);imgWidth = int(imgWidth)
    # 均值聚类提取前景:二维转一维
    imgVec = np.float32(srcImage.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret,label,clusCenter = cv2.kmeans(imgVec,2,None,criteria,10,flags)
    clusCenter = np.uint8(clusCenter)
    clusResult = clusCenter[label.flatten()]
    imgres = clusResult.reshape((srcImage.shape))
    imgres = cv2.cvtColor(imgres,cv2.COLOR_BGR2GRAY)
    bwThresh = int((np.max(imgres)+np.min(imgres))/2)
    _,thresh = cv2.threshold(imgres,bwThresh,255,cv2.THRESH_BINARY_INV)
    threshRotate = cv2.merge([thresh,thresh,thresh])
    # 确定前景外接矩形
    #find contours
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    minvalx = np.max([imgHeight,imgWidth]);maxvalx = 0
    minvaly = np.max([imgHeight,imgWidth]);maxvaly = 0
    maxconArea = 0;maxAreaPos = -1
    for i in range(len(contours)):
        if maxconArea < cv2.contourArea(contours[i]):
            maxconArea = cv2.contourArea(contours[i])
            maxAreaPos = i
    objCont = contours[maxAreaPos]
    # 旋转校正前景
    rect = cv2.minAreaRect(objCont)
    for j in range(len(objCont)):
        minvaly = np.min([minvaly,objCont[j][0][0]])
        maxvaly = np.max([maxvaly,objCont[j][0][0]])
        minvalx = np.min([minvalx,objCont[j][0][1]])
        maxvalx = np.max([maxvalx,objCont[j][0][1]])
    if rect[2] <= -45:
        rotAgl = 90 +rect[2]
    else:
        rotAgl = rect[2]
    if rotAgl == 0:
        panelImg = srcImage[minvalx:maxvalx,minvaly:maxvaly,:]
    else:
        rotCtr = rect[0]
        rotCtr = (int(rotCtr[0]),int(rotCtr[1]))
        rotMdl = cv2.getRotationMatrix2D(rotCtr,rotAgl,1)
        imgHeight,imgWidth = srcImage.shape[:2]
        #图像的旋转
        dstHeight = math.sqrt(imgWidth *imgWidth + imgHeight*imgHeight)
        dstRotimg = cv2.warpAffine(threshRotate,rotMdl,(int(dstHeight),int(dstHeight)))
        dstImage = cv2.warpAffine(srcImage,rotMdl,(int(dstHeight),int(dstHeight)))
        dstRotimg = cv2.cvtColor(dstRotimg,cv2.COLOR_BGR2GRAY)
        _,dstRotBW = cv2.threshold(dstRotimg,127,255,0)
        contours, hierarchy = cv2.findContours(dstRotBW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        maxcntArea = 0;maxAreaPos = -1
        for i in range(len(contours)):
            if maxcntArea < cv2.contourArea(contours[i]):
                maxcntArea = cv2.contourArea(contours[i])
                maxAreaPos = i
        x,y,w,h = cv2.boundingRect(contours[maxAreaPos])
        #提取前景：panel
        panelImg = dstImage[int(y):int(y+h),int(x):int(x+w),:]

    return panelImg

if __name__=="__main__":
    # file = '/home/linxu/Desktop/1_0_0_1020_1_0/状态指示器/Green/123.jpeg'
    # file = '/home/linxu/Documents/Work/testpic/数显表/1/image_180.jpg'
    file = '/home/hxzh02/文档/defectDetect/金属锈蚀(复件)/src/285.jpg'
    srcImage = cv2.imread(file)
    cv2.imshow('src', srcImage)
    a = panelAbstract(srcImage)
    cv2.imshow('figa',a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
