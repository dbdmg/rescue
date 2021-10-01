import torch
import numpy as np
from neural_net.utils import class_IU

from torch import nn

class AccuracyMulticlassStorage():
    def __init__(self, one_hot):
        self.one_hot = one_hot
        self.reset()
        self.sigm = nn.Sigmoid()

    def reset(self):
        self.always_0 = 0
        self.total = 0
        self.correct = 0
        self.intersection = 0
        self.union = 0

    def last(self):
        #print('        Always-0 predictor pixel accuracy: %f' % (self.always_0 / self.total))
        print('        Pixel accuracy: %f' % (self.correct / self.total))
        print('        IoU: %f' % ((self.intersection + 1) / (self.union + 1)))
        return

    def __call__(self, outputs, gt):
        if self.one_hot:
            gt = gt.argmax(dim=1, keepdim=True)
        if outputs.shape[1] > 1:
            outputs = outputs.argmax(dim=1, keepdim=True)
        elif outputs.shape[1] == 1:
            outputs = self.sigm(outputs)
            outputs = torch.where(outputs > 0.5, torch.tensor(1.0, device=outputs.device), torch.tensor(0.0, device=outputs.device))

        self.correct += (outputs == gt).sum().item()
        self.always_0 += (gt == 0).sum().item()
        self.total += gt.numel()
        
        intersection_now = (outputs * bin_gt).long().sum().item()
        self.intersection += intersection_now
        self.union += (outputs.long().sum().item() + bin_gt.long().sum().item() - intersection_now)
        return

class AccuracyBinStorage():
    def __init__(self, one_hot, device=None, threshold=0.5):
        self.reset()
        self.one_hot = one_hot
        self.threshold = threshold
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sigm = nn.Sigmoid()

    def reset(self):
        self.always_0 = 0
        self.total = 0
        self.correct = 0
        self.intersection = 0
        self.union = 0

    def last(self):
        #print('Always-0 predictor performance: %f' % (self.always_0 / self.total))
        print('        Pixel accuracy: %f' % (self.correct / self.total))
        print('        IoU: %f' % ((self.intersection + 1) / (self.union + 1)))
        return

    def __call__(self, outputs, gt):
        bin_gt = gt
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if self.one_hot:
            bin_gt = bin_gt.argmax(dim=1, keepdim=True)

        if outputs.shape[1] == 1:
            outputs = torch.where(self.sigm(outputs) >= self.threshold, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        else:
            outputs = outputs.argmax(dim=1, keepdim=True)
            
        bin_gt = torch.where(bin_gt >= 1.0, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        self.correct += (outputs == bin_gt).sum().item()
        self.always_0 += (bin_gt == 0).sum().item()
        self.total += bin_gt.numel()
        
        intersection_now = (outputs * bin_gt).long().sum().item()
        self.intersection += intersection_now
        self.union += (outputs.long().sum().item() + bin_gt.long().sum().item() - intersection_now)
        return



class AccuracySingleRegrStorage():
    def __init__(self):
        self.reset()

    def reset(self):
        self.always_0 = 0
        self.total = 0
        self.correct = 0
        self.intersection = 0
        self.union = 0

    def last(self):
        #print('Always-0 predictor performance: %f' % (self.always_0 / self.total))
        print('        Pixel accuracy: %f' % (self.correct / self.total))
        print('        IoU: %f' % ((self.intersection + 1) / (self.union + 1)))


    def __call__(self, outputs, gt):
        rounded = outputs.round()
        self.total += gt.numel()
        self.always_0 += (gt == 0).sum().item()
        self.correct += (gt == rounded).sum().item()
        classintersection, classunion = class_IU(gt=gt, prediction=rounded)
        self.intersection += classintersection
        self.union += classunion

class AccuracyRegrStorage():
    def __init__(self):
        self.reset()

    def reset(self):
        self.always_0 = 0
        self.total = 0
        self.correct = 0
        self.intersection = 0
        self.union = 0

    def last(self):
        #print('Always-0 predictor performance: %f' % (self.always_0 / self.total))
        print('        Pixel accuracy: %f' % (self.correct / self.total))
        print('        IoU: %f' % ((self.intersection + 1) / (self.union + 1)))


    def __call__(self, outputs, gt):
        rounded = outputs[1].round()
        self.total += gt.numel()
        self.always_0 += (gt == 0).sum().item()
        self.correct += (gt == rounded).sum().item()
        classintersection, classunion = class_IU(gt=gt, prediction=rounded)
        self.intersection += classintersection
        self.union += classunion

class AccuracyAllStorage():
    def __init__(self, one_hot, device=None, threshold=0.5):
        self.bin_storage = AccuracyBinStorage(one_hot, device, threshold)
        self.regr_storage = AccuracyRegrStorage()

    def reset(self):
        self.bin_storage.reset()
        self.regr_storage.reset()

    def last(self):
        print('    - Binary U-Net:')
        self.bin_storage.last()
        print('    - Regression U-Net:')
        self.regr_storage.last()

    def __call__(self, outputs, gt):
        self.bin_storage(outputs, gt)
        self.regr_storage(outputs, gt)