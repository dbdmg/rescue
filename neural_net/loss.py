import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5, weight=None, compact_data=True, self_compute_weight=False):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.class_weight = weight
        self.compact_data = compact_data
        self.self_compute_weight = self_compute_weight

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, value=1)


        nonlin_output = net_output
        if self.apply_nonlin is not None:
            nonlin_output = self.apply_nonlin(nonlin_output)

        my_in = self.flatten(nonlin_output)
        target = self.flatten(y_onehot)
        target = target.float()
        if self.self_compute_weight:
            target_sum = target.sum(-1)
            class_weights = 1. / (target_sum * target_sum).clamp(min=self.smooth)
            class_weights = class_weights.detach()

        if self.self_compute_weight:
            intersect = (my_in * target).sum(-1) * class_weights
        else:
            intersect = (my_in * target).sum(-1)
        if self.class_weight is not None:
            weight = self.class_weight.detach()
            intersect = weight * intersect
        if self.compact_data:
            intersect = intersect.sum()

        if self.self_compute_weight:
            denominator = ((my_in + target).sum(-1) * class_weights).sum()
        else:
            denominator = (my_in + target).sum(-1)
        if self.compact_data:
            denominator = denominator.sum()

        result = 1. - 2. * intersect / denominator.clamp(min=self.smooth)
        return result

    @classmethod
    def flatten(cls, tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
        (N, C, D, H, W) -> (C, N * D * H * W)
        """
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)

class IndexLoss(nn.Module):
    def __init__(self, index, gt_one_hot, loss, device=None):
        super(IndexLoss, self).__init__()
        assert index == 0 or index == 1
        self.index = index
        self.gt_one_hot = gt_one_hot
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = loss

    def forward(self, outputs, gt):
        out = outputs[self.index]
        if self.index == 0:
            bin_gt = gt
            if self.gt_one_hot:
                bin_gt = gt.argmax(dim=1, keepdim=True)
            bin_gt = torch.where(bin_gt >= 1.0, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))

            return self.loss(out, bin_gt)
        elif self.index == 1:
            return self.loss(out, gt)

# class F1Score(nn.Module):
#     def __init__(self, n_classes, smooth=1e-5):
#         super(F1Score, self).__init__()
#         self.n_classes = n_classes
#         self.smooth = smooth
#         return
#
#     def forward(self, outputs, gt):
#         out_scattered = self.scatter(outputs)
#         gt_scattered = self.scatter(gt)
#
#         num = (out_scattered * gt_scattered).sum(-1)
#         den = (out_scattered + gt_scattered).sum(-1).clamp(min=self.smooth)
#         result = 1 - 2 * (num / den)
#         return result.sum()
#
#     def scatter(self, t):
#         shp = (self.n_classes, t.shape[0])
#         t = t.long()
#         t.unsqueeze_(0)
#         result = torch.zeros(shp, device=t.device)
#         result.scatter_(0, t, value=1.)
#         return result

class F1Score(nn.Module):
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = 'weighted'):
        """
        average: averaging method, default micro
        """
        super(F1Score, self).__init__()
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor):
        """
        Calculate f1 micro.
            predictions: tensor with predictions
            labels: tensor with original labels
        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor, labels: torch.Tensor, label_id: int):
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = (torch.eq(labels, predictions) * torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return f1, true_count

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor):
        predictions = predictions.long().flatten()
        labels = labels.long().flatten()

        # simpler calculation for micro
        if self.average == 'micro':
            return 1 - self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(1, len(labels.unique()) + 1):
            f1, true_count = self.calc_f1_count_for_label(predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))

        return 1 - f1_score


class F1MSE(nn.Module):
    def __init__(self, f1=F1Score(), mse=nn.MSELoss()):
        super(F1MSE, self).__init__()
        self.f1 = f1
        self.mse = mse

    def forward(self, inputs, targets):
        mse = self.mse(inputs.float(), targets.float())
        f1 = self.f1(inputs, targets)

        return f1 * mse


class MyMSE(nn.Module):
    def __init__(self, n_classes, smooth=1e-5):
        super(MyMSE, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        return
    
    def forward(self, outputs, gt):
        gt_scattered = self.scatter(gt)
        out_scattered = self.scatter(outputs, gt)

        elem_counters = self.count_elements(gt).clamp(min=self.smooth)
        diff = gt_scattered - out_scattered
        diff = (diff ** 2).sum((0, *range(2, len(diff.shape))))
        mse = diff / elem_counters
        return  mse

    def count_elements(self, gt):
        with torch.no_grad():
            shp = (gt.shape[0], self.n_classes, *(gt.shape[2:]))
            counter = torch.zeros(shp, device=gt.device)
            gt = gt.long()
            counter.scatter_(1, gt, value=1.)
            result = counter.sum((0, *(range(2, len(gt.shape)))))
        return result

    def scatter(self, t, index=None):
        assert t.shape[1] == 1
        shp = (t.shape[0], self.n_classes, *(t.shape[2:]))
        result = torch.zeros(shp, device=t.device)
        if index is None:
            index = t
        index = index.long()
        t = t.float()
        result.scatter_(1, index, t)
        return result


class MSEDiceLoss(nn.Module):
    def __init__(self, mask_one_hot, n_classes, compute_weights):
        super(MSEDiceLoss, self).__init__()
        self.mask_one_hot = mask_one_hot
        self.n_classes = n_classes
        self.mymse = MyMSE(n_classes)
        self.dice = GDiceLossV2(compact_data=False, self_compute_weight=compute_weights)
        self.compute_weights = compute_weights
        return

    def forward(self, outputs, gt):
        if self.mask_one_hot:
            gt = gt.argmax(axis=1, keepdim=True)

        rounded_outputs = outputs.clamp(min=0, max=(self.n_classes - 1)).round().long()
        scattered_shape = (outputs.shape[0], self.n_classes, *(outputs.shape[2:]))
        out_scattered = torch.zeros(scattered_shape, device=outputs.device)
        out_scattered.scatter_(1, rounded_outputs, value=1.)
        dice_losses = self.dice(out_scattered, gt).float()

        # mse = []
        # for m in range(self.n_classes):
        #     mask = gt == m
        #     pred = outputs[mask].float()
        #     pred_rounded = rounded_outputs[mask].float()
        #     real = gt[mask].float()
        #     mse.append(self.mse(pred, real))

        mymse = self.mymse(outputs, gt)
        result = mymse * dice_losses
        return  result.sum()


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs) #is it necessary also for multiclass?

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        if inputs.max().item() > 1: #assumed to be in regression case
            print("IoULoss: using multiclass")
            rounded = inputs.round()
            intersection, union = class_IU(rounded, targets, get_tensor=True)
        else:
            # intersection is equivalent to True Positive count
            # union is the mutually inclusive area of all labels & predictions
            intersection = (inputs * targets).sum()
            total = (inputs + targets).sum()
            union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU
    
    
class FuzzyIoULoss(nn.Module):
    def __init__(self):
        """
        Now only used for binary case
        """
        super(FuzzyIoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # flatten and rescale label and prediction tensors
        inputs = (inputs).view(-1)
        targets = (targets).view(-1)

        # Sigmoid is unnecessary in binary case
        if targets.max() > 1:
            inputs = torch.sigmoid(inputs)
            targets = torch.sigmoid(targets)
        
        intersection_vect = (inputs * targets)
        
        union = (inputs + targets - intersection_vect).sum()
        intersection = intersection_vect.sum()

        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU


class ComboLoss(nn.Module):
    def __init__(self, loss1=GDiceLossV2(), loss2=nn.BCEWithLogitsLoss()):
        super(ComboLoss, self).__init__()
        self.loss1=eval(loss1)() if type(loss1) == str else loss1
        self.loss2=eval(loss2)() if type(loss2) == str else loss2

    def forward(self, inputs, targets, alpha=.5):
        combo =  alpha * self.loss1(inputs, targets) + (1 - alpha) * self.loss2(inputs, targets)
        return combo


class softIoULoss(nn.Module):
    def __init__(self, eps=0.5):
        """
        Used to reflect the idea of IoU on the regression case before discretization in classes.
        Always use ground truth as second argument
        """
        super(softIoULoss, self).__init__()
        self.eps = eps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets, smooth=1):
        intersection = torch.tensor(0.0, device=self.device)
        union = torch.tensor(0.0, device=self.device)

        # flatten and rescale label and prediction tensors
        inputs = (inputs).view(-1)
        targets = (targets).view(-1)
        n_classes = targets.max().int()
        for c in range(0, n_classes+1):
            class_mask = torch.where(targets == c, torch.tensor(1.0, device=self.device),
                                     torch.tensor(0.0, device=self.device))

            invsigm_distance = torch.sigmoid(- torch.abs(inputs - targets) + self.eps)
            tmp_intersection = invsigm_distance * class_mask
            intersection += tmp_intersection.sum()
            union += (class_mask + invsigm_distance - tmp_intersection).sum()
        sIoU = (intersection + smooth) / (union + smooth)
        return 1-sIoU

