'''
- 感知哈希（Perceptual Hash, pHash）:
    优点：pHash 可以在图片有轻微修改（如缩放、裁剪、旋转等）的情况下，仍然得到相同或者相似的哈希值。这使得 pHash 适合用于检测内容相似的图片。
    缺点：pHash 的计算相对复杂，可能需要更长的时间。另外，如果两张图片的内容虽然相似，但在结构上有较大的不同（如不同的视角、不同的排列顺序等），pHash 可能无法检测出它们的相似性。
- 平均哈希（Average Hash, aHash）:
    优点：aHash 的计算简单快速，适合用于大规模的图片处理任务。与 pHash 类似，aHash 也可以在图片有轻微修改的情况下，得到相同或者相似的哈希值。
    缺点：aHash 对图片的细节变化敏感度较低，可能无法检测出图片内容的微小变化。
- 差异哈希（Difference Hash, dHash）:
    优点：dHash 的计算简单快速，适合用于大规模的图片处理任务。dHash 可以检测出图片内容的微小变化，适合用于检测内容几乎一样，但有微小差异的图片。
    缺点：dHash 对图片的全局变化（如缩放、裁剪、旋转等）敏感度较低，可能无法检测出这些变化。
- CNN 方法：
    优点：CNN 方法可以学习图片的高级特征，因此可以更准确地检测出内容相似的图片，即使这些图片在视觉上有较大的差异。
    缺点：CNN 方法的计算复杂度高，需要更长的时间和更多的计算资源。另外，CNN 方法需要预训练的模型，对于没有预训练模型的特定任务，CNN 方法可能无法得到好的结果。
根据您的需求，您可以选择最适合的方法。例如，如果您需要快速处理大量图片，可以选择 aHash 或 dHash。如果您需要检测内容相似，但在视觉上有差异的图片，可以选择 pHash 或 CNN 方法。
'''

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