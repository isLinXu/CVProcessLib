import os
from PIL import Image
from imagehash import phash
from tqdm import tqdm


def get_image_phash(image_path):
    with Image.open(image_path) as img:
        image_phash = phash(img)
    return image_phash


def remove_duplicate_images(folder_path, hash_diff_threshold=5):
    image_phashes = {}
    duplicate_images = []

    file_list = os.listdir(folder_path)
    for filename in tqdm(file_list, desc="处理图片", unit="张"):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            Image.open(file_path).verify()  # 检查文件是否为有效的图片
        except (IOError, SyntaxError) as e:
            print(f"无法读取图片文件: {file_path}")
            continue

        image_phash = get_image_phash(file_path)

        is_duplicate = False
        for stored_phash in image_phashes.values():
            if abs(image_phash - stored_phash) <= hash_diff_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            duplicate_images.append(file_path)
            os.remove(file_path)
            print(f"删除重复图片: {file_path}")
        else:
            image_phashes[filename] = image_phash

    return duplicate_images


folder_path = "path/to/your/image/folder"
hash_diff_threshold = 5  # 允许的哈希差异阈值，值越小表示图片越相似
duplicate_images = remove_duplicate_images(folder_path, hash_diff_threshold)
print(f"共删除 {len(duplicate_images)} 张重复图片")