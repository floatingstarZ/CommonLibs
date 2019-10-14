import numpy as np
import torch
from commonlibs.transform_tools.type_transform import to_bbox_type

# three types of bbox representation
# 1. coco style(coco): x1, y1, h, w
# 2. center style(center): xc, yc, h, w
# 3. corner style(corner): x1, y1, x2, y2

# first transform to numpy stype array
# list -> array

def split_bboxes(bboxes):
    """
    :param bboxes: any style 
    :return: 
    """
    bboxes = to_bbox_type(bboxes)
    if bboxes.shape[1] == 0:
        return bboxes
    return (bboxes[:, 0:1], bboxes[:, 1:2],
            bboxes[:, 2:3],  bboxes[:, 3:4])

def coco2center(bboxes):
    """
    :param bbox: N x 4 (coco style)
    :return: bbox: N x 4 (center style)
    """
    splited = split_bboxes(bboxes)
    if not isinstance(splited, tuple):
        return bboxes
    (x1, y1, h, w) = splited
    return torch.cat([x1+h/2, y1+w/2, h, w], dim=1)

def coco2corner(bboxes):
    """
    :param bboxes: N x 4 (coco style)
    :return: bbox: N x 4 (corner style)
    """
    splited = split_bboxes(bboxes)
    if not isinstance(splited, tuple):
        return bboxes
    (x1, y1, h, w) = splited
    return torch.cat([x1, y1, x1+h, y1+w], dim=1)

def center2coco(bboxes):
    """
    :param bboxes: N x 4 (center style)
    :return: bbox: N x 4 (coco style)
    """
    splited = split_bboxes(bboxes)
    if not isinstance(splited, tuple):
        return bboxes
    (xc, yc, h, w) = splited
    return torch.cat([xc-h/2, yc-w/2, h, w], dim=1)

def center2corner(bboxes):
    """
    :param bbox: N x 4 (center style)
    :return: bbox: N x 4 (corner style)
    """
    splited = split_bboxes(bboxes)
    if not isinstance(splited, tuple):
        return bboxes
    (xc, yc, h, w) = splited
    return torch.cat([xc-h/2, yc-w/2, xc+h/2, yc+w/2], dim=1)

def corner2center(bboxes):
    """
    :param bboxes: N x 4 (corner style)
    :return: bbox: N x 4 (center style)
    """
    splited = split_bboxes(bboxes)
    if not isinstance(splited, tuple):
        return bboxes
    (x1, y1, x2, y2) = splited
    return torch.cat([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=1)

def corner2coco(bboxes):
    """
    :param bboxes: N x 4 (corner style)
    :return: bbox: N x 4 (coco style)
    """
    splited = split_bboxes(bboxes)
    if not isinstance(splited, tuple):
        return bboxes
    (x1, y1, x2, y2) = splited
    return torch.cat([x1, y1, x2-x1, y2-y1], dim=1)




if __name__ == '__main__':
    a = [1, 1, 10, 10]
    print(coco2center(a))
    print(coco2corner(a))
    print(center2coco(a))
    print(center2corner(a))
    print(corner2center(a))
    print(corner2coco(a))


