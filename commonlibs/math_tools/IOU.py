import torch

def singleIOU(gt, bboxes):
    """
    
    :param gt: left top right down
    :param bboxes: N * 4
    :return: 
    """
    [x1, y1, x2, y2] = gt
    inter_lt = (gt[0].max(bboxes[:, 0]), gt[1].max(bboxes[:, 1]))
    inter_rd = (gt[2].min(bboxes[:, 2]), gt[3].min(bboxes[:, 3]))
    z = torch.Tensor([0.0])
    inter_w = (inter_rd[0] - inter_lt[0]).max(z)
    inter_h = (inter_rd[1] - inter_lt[1]).max(z)
    inter_area = inter_w * inter_h
    area_gt = ((x2 - x1)*(y2 - y1)).max(z)
    if area_gt <= 0:
        return torch.zeros(bboxes.shape[0])
    area_bboxes = ((bboxes[:, 2] - bboxes[:, 0]) * \
                  (bboxes[:, 3] - bboxes[:, 1])).max(z)
    IOU = inter_area / (area_gt + area_bboxes - inter_area)
    return IOU

def IOU(gts, bboxes):
    """
    
    :param gts: M * 4
    :param bboxes: N * 4
    :return: M * N
    """
    IOU = []
    for gt in gts:
        IOU.append(singleIOU(gt, bboxes).reshape(1, -1))
    return torch.cat(IOU, dim=0)



if __name__ == '__main__':
    gts = torch.Tensor([[3, 3, 6, 6],
                       [0, 0, 1, 1],
                       [1, 1, 1, 1]])
    bboxes = torch.Tensor([[3,3,6,6],
                           [4,4,7,7],
                           [2,2,5,5],
                           [3,2,6,5]])
    print(singleIOU(gts[0], bboxes))
    print(singleIOU(torch.Tensor([0, 0, 1, 1]), bboxes))
    print(singleIOU(torch.Tensor([1, 1, 1, 1]), bboxes))
    print(IOU(gts, bboxes))
























