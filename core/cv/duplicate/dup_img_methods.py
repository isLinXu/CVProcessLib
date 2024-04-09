from imagededup.methods import PHash, AHash, DHash, CNN
from pathlib import Path
import os

def remove_duplicate_images(folder_path, method='phash', score_threshold=1.0):
    if method == 'phash':
        hasher = PHash()
    elif method == 'ahash':
        hasher = AHash()
    elif method == 'dhash':
        hasher = DHash()
    elif method == 'cnn':
        hasher = CNN()
    else:
        raise ValueError("无效的去重方法，请选择 'phash', 'ahash', 'dhash' 或 'cnn'")

    # 找到重复的图片
    duplicates = hasher.find_duplicates(image_dir=folder_path, scores=True)

    # 删除重复的图片
    num_duplicates = hasher.delete_duplicates(duplicates, folder_path, score_threshold)
    print(f"共删除 {num_duplicates} 张重复图片")

folder_path = "path/to/your/image/folder"
method = 'phash'  # 选择去重方法：'phash', 'ahash', 'dhash' 或 'cnn'
score_threshold = 1.0  # 相似度分数阈值，范围为 0 到 1，值越大表示图片越相似
remove_duplicate_images(folder_path, method=method, score_threshold=score_threshold)