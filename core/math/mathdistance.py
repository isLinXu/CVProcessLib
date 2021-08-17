import numpy as np



def get_euclidean_distance(img,_pixel, _center):
    """
    获得欧氏距离
    :param _pixel: 像素
    :param _center: 中心
    :return: 欧氏距离
    """
    # get channels count
    img_shape = img.shape
    n_channels = img_shape[1]
    d_pow_2 = 0
    for _channel_index in range(n_channels):
        d_pow_2 += pow(_pixel[_channel_index] - _center[_channel_index], 2)
    return np.sqrt(d_pow_2)


def get_nearest_center(_pixel,centers,k):
    """
    获得最近的中心
    :param _pixel: 像素
    :return: 最近的中心的index
    """
    min_center_d = get_euclidean_distance(_pixel, centers[0])
    min_center_index = 0
    for _center_index in range(1, k):
        d = get_euclidean_distance(_pixel, centers[_center_index])
        if d < min_center_d:
            min_center_d = d
            min_center_index = _center_index
    return min_center_index
