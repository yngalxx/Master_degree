import csv
import numpy as np

def from_tsv_to_list(path, delimiter="\n"):
    tsv_file = open(path)
    read_tsv = csv.reader(tsv_file, delimiter=delimiter)
    expected = list(read_tsv)

    return [item for sublist in expected for item in sublist]


def collate_fn(batch):
    return tuple(zip(*batch))


# def iou_list(bbox_true, bbox_pred):
#     bbox_list_of_tuples = zip(bbox_true, bbox_pred)
#     iou_list = []
#     for boxA, boxB in range(len(bbox_list_of_tuples)):
#         # determine the (x, y)-coordinates of the intersection rectangle
#         xA = max(boxA[0], boxB[0])
#         yA = max(boxA[1], boxB[1])
#         xB = min(boxA[2], boxB[2])
#         yB = min(boxA[3], boxB[3])
#         # compute the area of intersection rectangle
#         interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#         # compute the area of both the prediction and ground-truth
#         # rectangles
#         boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#         boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#         # compute the intersection over union by taking the intersection
#         # area and dividing it by the sum of prediction + ground-truth
#         # areas - the interesection area
#         iou = interArea / float(boxAArea + boxBArea - interArea)
#         iou_list.append(iou)

#     return iou_list