import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_feature_extractor(feature_extractor='ORB'):
    if feature_extractor == 'ORB':
        extractor = cv2.ORB_create()
    elif feature_extractor == 'SIFT':
        extractor = cv2.SIFT_create()
    elif feature_extractor == 'SURF':
        extractor = cv2.xfeatures2d.SURF_create()
    elif feature_extractor == 'AKAZE':
        extractor = cv2.AKAZE_create()
    else:
        raise ValueError("Invalid feature_extractor. Choose from 'ORB', 'SIFT', 'SURF', 'AKAZE'.")
    return extractor


def create_matcher(feature_extractor='ORB', matcher='BF'):
    if matcher == 'BF':
        if feature_extractor in ['ORB', 'AKAZE']:
            norm_type = cv2.NORM_HAMMING
        else:
            norm_type = cv2.NORM_L2
        match = cv2.BFMatcher(norm_type, crossCheck=True)
    elif matcher == 'FLANN':
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        match = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError("Invalid matcher. Choose from 'BF', 'FLANN'.")
    return match


def compute_similarity(img1, img2, feature_extractor='ORB', matcher='BF', distance_threshold=40, draw_matches=False):
    # 检查图像是否为空
    if img1 is None or img2 is None:
        raise ValueError("Could not open or find the images!")

    # 检查图像是否为灰度图像，如果不是，转换为灰度图像
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 创建特征提取器
    extractor = create_feature_extractor(feature_extractor)

    # 提取关键点和描述符
    kp1, des1 = extractor.detectAndCompute(img1, None)
    kp2, des2 = extractor.detectAndCompute(img2, None)

    # 创建匹配器
    match = create_matcher(feature_extractor, matcher)

    # 匹配描述符
    matches = match.match(des1, des2)

    # 检查是否有匹配结果
    if not matches:
        raise ValueError("No matches found!")

    # 筛选匹配点
    good_matches = [m for m in matches if m.distance < distance_threshold]

    # 计算相似度
    similarity = len(good_matches) / len(matches)

    # 绘制匹配结果（可选）
    if draw_matches:
        result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 格式转换为 RGB 格式
        plt.imshow(result)
        plt.show()

    return similarity


if __name__ == "__main__":
    # 读取图像
    img1 = cv2.imread('banana1.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('banana3.jpg', cv2.IMREAD_COLOR)
    # 计算相似度
    similarity = compute_similarity(img1, img2, feature_extractor='ORB', matcher='BF',
                                    distance_threshold=40,
                                    draw_matches=True)
    print("相似度:", similarity)