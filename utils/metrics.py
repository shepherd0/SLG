import torch
import torch.nn.functional as F

"""
The evaluation implementation refers to the following paper:
"Selective Feature Aggregation Network with Area-Boundary Constraints for Polyp Segmentation"
https://github.com/Yuqi-cuhk/Polyp-Seg
"""
def evaluate(pred, gt):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    # TP—— 预测为 P （正例）, 预测对了， 本来是正样本，检测为正样本（真阳性）。
    TP = pred_binary.mul(gt_binary).sum()
    # F P—— 预测为 P （正例）, 预测错了， 本来是负样本，检测为正样本（假阳性）。
    FP = pred_binary.mul(gt_binary_inverse).sum()
    # TN—— 预测为 N （负例）, 预测对了， 本来是负样本，检测为负样本（真阴性）。
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    # F N—— 预测为 N （负例）, 预测错了， 本来是正样本，检测为负样本（假阴性）。
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP = torch.Tensor([1]).cuda()

    # recall
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    # Specificity = TN / (TN + FP)

    FPR = FP / (FP + TN)

     # Precision or positive predictive value
    Precision = TP / (TP + FP)

    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)
    # F1 = 2 * TP / (FP + 2 * TP + FN)
    # F2 score
    # F2 = 5 * Precision * Recall / (4 * Precision + Recall)

    # ZSI
    ZSI = 2 * TP / (2 * TP + FP + FN)

    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)

    # IoU for poly
    IoU_poly = TP / (TP + FP + FN)

    # IoU for background
    IoU_bg = TN / (TN + FP + FN)

    # mean IoU
    IoU_mean = (IoU_poly + IoU_bg) / 2.0



    return Recall, FPR, Precision, F1, ZSI,  ACC_overall, IoU_poly, IoU_bg, IoU_mean




class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics
