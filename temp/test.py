

if __name__ == '__main__':
    # coding=utf-8
    # 导入一些python包
    from imutils.perspective import four_point_transform
    from imutils import contours
    import imutils
    import cv2

    # 定义每一个数字对应的字段
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 1, 0): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }
    path = '/home/linxu/Documents/Work/sources/textDetect_project/cut_out/1/image_0.jpg'

    # 读取输入图片
    image = cv2.imread(path)

    # 将输入图片裁剪到固定大小
    # image = imutils.resize(image, height=500)
    # 将输入转换为灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行高斯模糊操作
    blurred = cv2.GaussianBlur(gray, (1, 1), 10)

    # 执行边缘检测
    edged = cv2.Canny(blurred, 0, 255, 255)
    # cv2.imwrite('edge.png', edged)
    cv2.imshow('edged',edged)

    # 在边缘检测map中发现轮廓
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # 根据大小对这些轮廓进行排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    # 循环遍历所有的轮廓
    for c in cnts:
        # 对轮廓进行近似
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 如果当前的轮廓有4个顶点，我们返回这个结果，即LCD所在的位置
        if len(approx) == 4:
            displayCnt = approx
            break

    # 应用视角变换到LCD屏幕上
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    cv2.imshow('warped',warped)
    # cv2.imwrite('warped.png', warped)
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    # 使用阈值进行二值化
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imshow('thresh', thresh)
    # cv2.imwrite('thresh1.png', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # 使用形态学操作进行处理
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite('thresh2.png', thresh)
    cv2.waitKey()
    # 在阈值图像中查找轮廓，然后初始化数字轮廓列表
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []

    # 循环遍历所有的候选区域
    for c in cnts:
        # 计算轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(c)

        # 如果当前的这个轮廓区域足够大，它一定是一个数字区域
        if w >= 15 and (h >= 30 and h <= 40):
            digitCnts.append(c)

    # 从左到右对这些轮廓进行排序
    cnts, digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    digits = []

    # 循环处理每一个数字
    i = 0
    for c in digitCnts:
        # 获取ROI区域
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]

        # 分别计算每一段的宽度和高度
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)

        # 定义一个7段数码管的集合
        segments = [
            ((0, 0), (w, dH)),  # 上
            ((0, 0), (dW, h // 2)),  # 左上
            ((w - dW, 0), (w, h // 2)),  # 右上
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # 中间
            ((0, h // 2), (dW, h)),  # 左下
            ((w - dW, h // 2), (w, h)),  # 右下
            ((0, h - dH), (w, h))  # 下
        ]
        on = [0] * len(segments)

        # 循环遍历数码管中的每一段
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):  # 检测分割后的ROI区域，并统计分割图中的阈值像素点
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)

            # 如果非零区域的个数大于整个区域的一半，则认为该段是亮的
            if total / float(area) > 0.5:
                on[i] = 1

        # 进行数字查询并显示结果
        digit = DIGITS_LOOKUP[tuple(on)]
        digits.append(digit)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(output, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # 显示最终的输出结果
    print(u"{}{}.{} \u00b0C".format(*digits))
    cv2.imshow("Input", image)
    cv2.imshow("Output", output)
    cv2.waitKey(0)