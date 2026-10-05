import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric


def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)
    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    recall = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TP : True Positive
        # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2#（1,1）
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2#（0,1）
    recall = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return recall

def get_specificity(SR,GT,threshold=0.5):
    Specificity = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
        # TN : True Negative
        # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2#（0,0）
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2#（1,0）
    Specificity = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return Specificity

def get_precision(SR,GT,threshold=0.5):
    Precision = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
        # TP : True Positive
        # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    Precision = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return Precision

def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    has_output = np.any(output_)
    has_target = np.any(target_)

    if not has_output or not has_target:
        hd95 = np.nan
    else:
        hd95 = metric.binary.hd95(output_, target_)

    output_ = torch.tensor(output_)
    target_=torch.tensor(target_)
    recall = get_sensitivity(output_,target_,threshold=0.5)
    precision = get_precision(output_,target_,threshold=0.5)
    specificity= get_specificity(output_,target_,threshold=0.5)
    acc=get_accuracy(output_,target_,threshold=0.5)
    F1 = 2*recall*precision/(recall+precision + 1e-6)


    return iou, dice , recall, precision, F1,specificity,acc, hd95





