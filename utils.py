import numpy as np
import torch

def iou(box,boxes,isMin = False):
    box_area = (box[2]-box[0])*(box[3]-box[1])#[x1,y1,x2,y2,c]
    boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    xx1 = np.maximum(box[0],boxes[:,0])
    yy1 = np.maximum(box[1], boxes[:,1])
    xx2 = np.minimum(box[2],boxes[:,2])
    yy2 = np.minimum(box[3],boxes[:,3])

    w = np.maximum(0,xx2-xx1)
    h = np.maximum(0,yy2-yy1)

    inter = w*h
    if isMin:
        #最小面积
        over = np.true_divide(inter,np.minimum(box_area,boxes_area))
    else:
        # 并集
        over = np.true_divide(inter, (box_area+boxes_area-inter))
    return over


def nms(boxes,thresh,isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])
    _boxes = boxes[(-boxes[:,4]).argsort()]
    r_boxes = []
    while _boxes.shape[0] >1:
        a_box = _boxes[0]
        b_box = _boxes[1:]

        r_boxes.append(a_box)

        index = np.where(iou(a_box,b_box,isMin) < thresh)
        _boxes = b_box[index]
    if _boxes.shape[0] >0:
        r_boxes.append(_boxes[0])
    return np.stack(r_boxes)

def convert_to_square(bbox):
    squre_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:,3]-bbox[:,1]
    w = bbox[:,2]-bbox[:,0]
    max_side = np.maximum(w,h)
    squre_bbox[:,0] = bbox[:,0]+w*0.5-max_side*0.5
    squre_bbox[:,1] = bbox[:, 1] + h* 0.5 - max_side * 0.5
    squre_bbox[:,2] = squre_bbox[:,0]+max_side
    squre_bbox[:,3] = squre_bbox[:,1]+max_side

    return squre_bbox


def iou_torch(box, boxes, isMin=False, device='cuda'):
    '''
    计算box与boxes中的交并比
    :param box: 选中区域
    :param boxes: 其他区域列表
    :param isMin: 是否除以最小面积，否则除以并集
    :return: 选中区域与每个其他区域的交并比
    '''
    # 计算各个区域面积
    area = (box[2]-box[0]) * (box[3]-box[1])
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])

    # 获取交集区域的坐标点
    x1 = torch.max(boxes[:,0], box[0])
    y1 = torch.max(boxes[:,1], box[1])
    x2 = torch.min(boxes[:,2], box[2])
    y2 = torch.min(boxes[:,3], box[3])

    # 计算交集区域的长宽，并排除完全不相交的两区域，防止负负得正的面积
    w = torch.max(torch.zeros(x1.shape).to(device), x2-x1)
    h = torch.max(torch.zeros(x1.shape).to(device), y2-y1)

    # 计算交集区域面积
    area_intersection  = w*h

    # 判断是交集除以最小面积，还是交集除以并集
    if isMin:
        area_union = torch.min(area, areas)
    else:
        area_union = areas + area - area_intersection

    return torch.div(area_intersection, area_union)

def nms_torch(boxes, threshold=0.3, isMin=False):
    '''
    对输入的所有区域做非极大值抑制
    :param boxes: 所有区域
    :param threshold: 阈值，小于该阈值保留
    :param isMin: 是否除以最小面积，否则除以并集
    :return: 筛选后的区域
    '''
    # 二维数组，获取长度最好使用该方式，而不是len
    if boxes.shape[0] == 0:
        return torch.Tensor()

    out = []
    _boxes = boxes[torch.argsort(-boxes[:, -1])]    # 注意负号，从大到小排列

    while _boxes.shape[0] > 1:
        out.append(_boxes[0])
        # iou计算的是第一个和后边的交并比，因此返回的是除了第一个的索引序列，需要将第一个切除
        _boxes = _boxes[1:][torch.le(iou_torch(_boxes[0], _boxes[1:], isMin), threshold)]

    # 防止最后一个计算iou比较小，需要加入，但是while判定条件不符合退出循环，所以需要判断是否加上
    if _boxes.shape[0] > 0:
        out.append(_boxes[0])

    return torch.stack(out)    # 将列表中的ndarray整合为ndarray

def rect_to_square_torch(boxes):
    '''
    将矩形框转换为方形框。
    方式：按照长边将矩形边界扩充为正方形边界
    :param boxes: 矩形边界
    :return: 正方形边界
    '''
    if boxes.shape[0] == 0:
        return torch.Tensor()

    square_box = boxes.clone()

    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]

    max_side = torch.max(w, h)
    max_side_half = 0.5 * max_side

    # 每个点先移动到中间，然后减去长边的一半
    square_box[:,0] = boxes[:,0] + 0.5*w - max_side_half
    square_box[:,1] = boxes[:,1] + 0.5*h - max_side_half
    # 做上点加长边，即正方形边长
    square_box[:,2] = square_box[:,0] + max_side
    square_box[:,3] = square_box[:,1] + max_side

    return square_box