import numpy as np
import torch
import inspect

def all_to_numpy(data):
    pass
    # if isinstance(data)

def to_bbox_type(bboxes, dtype=torch.FloatTensor):
    """
    :param bboxes: list, numpy, tensor
    :return: bbox: n x m, all to float tensor
    """
    bboxes = dtype(bboxes)
    shape = bboxes.shape
    assert len(shape) <= 2

    if len(shape) == 1:
        bboxes = bboxes.unsqueeze(0)
    return bboxes


if __name__ == '__main__':
    # pass
    # # test to_bbox_type
    a = torch.Tensor([1,2, 3, 4])
    print(to_bbox_type(a))
    a = torch.Tensor([[1,2, 3, 4]])
    print(to_bbox_type(a))
    a = np.array([1,2, 3, 4])
    print(to_bbox_type(a))
    a = np.array([[1,2, 3, 4]])
    print(to_bbox_type(a))
    a = np.array([])
    print(to_bbox_type(a))
    a = [1, 2, 3, 4]
    print(to_bbox_type(a))
    a = []
    print(to_bbox_type(a).shape)
    a = [[1,2,3,4]]
    print(to_bbox_type(a))
