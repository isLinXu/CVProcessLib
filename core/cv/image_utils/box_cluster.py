'''
box聚类

Author: alex
Created Time: 2020年07月02日 星期四 17时09分54秒
'''
from sklearn.cluster import DBSCAN
from image_utils.line import iou_line


def boxes_cluster(boxes, row=True, col=True, iou_score=0.5):
    """将box框进行聚类
    :param boxes List[List[float]] 待聚类的box列表，每个box的格式：[x1,y1,x2,y2]
    :param row bool 是否进行行聚类
    :param col bool 是否进行列聚类
    :return row_labels List[int]|None 每个box的行id
    :return col_labels List[int]|None 每个box的列id
    """
    row_labels = None
    if row:
        row_labels = row_cluster(boxes, iou_score)

    col_labels = None
    if col:
        col_labels = col_cluster(boxes, iou_score)

    return row_labels, col_labels


def row_cluster(boxes, iou_score):
    """按行聚类"""
    boxes = [[box[1], box[3]] for box in boxes]
    return do_cluster(boxes, iou_score)


def col_cluster(boxes, iou_score):
    """按列聚类"""
    boxes = [[box[0], box[2]] for box in boxes]
    return do_cluster(boxes, iou_score)


def do_cluster(boxes, iou_score):
    """实际聚类"""
    # boxes = sorted(boxes, key=lambda x: x[0])
    db = DBSCAN(eps=iou_score, min_samples=1, metric=distance).fit(boxes)
    return db.labels_


def distance(box1, box2):
    """距离函数"""
    return 1 - iou_line(box1, box2)


if __name__ == '__main__':
    boxes = [
        [1, 1, 10, 10], [12, 2, 21, 12],
        [2, 9, 11, 20], [10, 11, 20, 22],
        [1.5, 22, 11.5, 30],
        [2, 39, 11, 50], [10, 41, 20, 52], [22, 40, 30, 51],
    ]
    row_labels, col_labels = boxes_cluster(boxes)
    print(row_labels)
    print(col_labels)
    assert row_labels.tolist() == [0, 0, 1, 1, 2, 3, 3, 3]
    assert col_labels.tolist() == [0, 1, 0, 1, 0, 0, 1, 2]
