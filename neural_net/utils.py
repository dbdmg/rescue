import os
import torch
import random
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple

from sklearn.metrics import mean_squared_error

from torch import nn

from neural_net.pspnet import PSPNet
import wandb
import argparse

from torch import nn, optim

from neural_net.unet import UNet
from neural_net.pspnet import PSPNet
from neural_net.nested_unet import NestedUNet
from neural_net.segnet import SegNet
from neural_net.attention_unet import AttentionUnet

from neural_net.loss import IoULoss, FuzzyIoULoss, GDiceLossV2, ComboLoss, softIoULoss, F1MSE


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
def eval_object(hdict, **default_kwargs):
    kwargs = hdict.copy()
    ob_type = kwargs.pop("name")
    
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)
        
    return eval(ob_type)(**kwargs)

from neural_net.concat_model import ConcatenatedModel


def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def produce_report(cm, result_path, test_set=None, sq_err=None, num_el=None, cm2=None):
    def _write_cm(f, cm):
        f.write('__________________________________________________________________________________________\n')
        f.write('Labels order (left-to-right, top-to-bottom): %s\n' % list(range(cm.shape[0])))
        f.write('__________________________________________________________________________________________\n')
        f.write('Confusion matrix:\n')
        f.write(str(cm) + '\n')
        f.write('Rows = ground truth\n')
        f.write('Columns = prediction\n')
        f.write('__________________________________________________________________________________________\n')
        total_per_class = cm.sum(axis=1)
        total = cm.sum()
        for x in range(cm.shape[0]):
            f.write('Total number of pixels in class %d: %d - %.3f (percentage)\n' % (x, total_per_class[x], (total_per_class[x] / total)))
        f.write('__________________________________________________________________________________________\n')
        f.write('Performances:\n')
        prec, rec, f1, acc = compute_prec_recall_f1_acc(cm)
        f.write('Overall accuracy: %.4f\n' % acc)
        f.write('###############\n')
        for x in range(cm.shape[0]):
            f.write('Class %d:\n' % x)
            f.write('Precision: %.4f\n' % prec[x])
            f.write('Recall: %.4f\n' % rec[x])
            f.write('f1: %.4f\n' % f1[x])
            f.write('###############\n')

        return

    with open(result_path, 'w') as f:
        if test_set is not None:
            f.write('Test set:')
            for el in test_set:
                f.write(el + ', ')
            f.write('\n')

        f.write('Confusion matrix #1\n')
        _write_cm(f, cm)
        if cm2 is not None:
            f.write('Confusion matrix #2\n')
            _write_cm(f, cm2)

        if sq_err is not None and num_el is not None:
            results = np.sqrt(sq_err / num_el)
            f.write('__________________________________________________________________________________________\n')
            f.write('RMSE on entire test set: %f\n' % results[-1])
            f.write('###############\n')
            for x in range(sq_err.size - 1):
                f.write('Class %d:\n' % x)
                f.write('RMSE: %f\n' % results[x])
                f.write('###############\n')

def compute_squared_errors(prediction, ground_truth, n_classes, check=False):
    """
    Separately for each class, compute total squared error (sq_err_res), and total count (count_res)
    
    Returns:
    -----
    (tuple)
    sq_errors: np.array([sq_err_class0, sq_err_class1,
               ..., total_sq_Err_all_classes])
               
    counts: np.array([n_pixels_class0, n_pixels_class1,
            ..., total_pixels])
    """

    squared_errors = [] # squared error for each class
    counters = []       # number of ground-truth pixels for each class

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().detach().cpu().numpy()

    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.squeeze().detach().cpu().numpy()

    if len(ground_truth.shape) == 3 and ground_truth.shape[-1] == 1:
        ground_truth = ground_truth.squeeze(axis=-1)

    if len(prediction.shape) == 3 and prediction.shape[-1] == 1:
        prediction = prediction.squeeze(axis=-1)

    mse_check = []

    # For each class label
    for idx in range(n_classes):
        mask = ground_truth == idx
        pred_data = prediction[mask]     # Predicted pixels corresponding to ground truth elements of class idx
        gt_data = ground_truth[mask]     # Ground truth pixels with class idx
        sq_err = np.square(pred_data - gt_data).sum()   # Squared error for those pixels
        n_elem = mask.sum()                             # Number of considered pixels

        squared_errors.append(sq_err)
        counters.append(n_elem)

        if check:
            if n_elem > 0:
                mse_check.append(mean_squared_error(gt_data, pred_data))
            else:
                mse_check.append(0)

    sq_err = np.square((prediction - ground_truth).flatten()).sum()     # Total squared error (all classes)
    if check:
        mse_check.append(mean_squared_error(ground_truth.flatten(), prediction.flatten()))
    n_elem = prediction.size
    squared_errors.append(sq_err) # [sq_err_class0, sq_err_class1,..., total_sq_Err_all_classes]
    counters.append(n_elem)       # total n. pixels

    sq_err_res = np.array(squared_errors)
    count_res = np.array(counters)

    if check:
        mymse = sq_err_res / count_res.clip(min=1e-5)
        mymse[np.isnan(mymse)] = 0

        mse_check = np.array(mse_check)
        assert (np.abs(mymse - mse_check) < 1e-6).all()

    return sq_err_res, count_res


def compute_prec_recall_f1_acc(conf_matr):
    accuracy = np.trace(conf_matr) / conf_matr.sum()

    predicted_sum = conf_matr.sum(axis=0)   # sum along column
    gt_sum = conf_matr.sum(axis=1)          # sum along rows
                
    diag = np.diag(conf_matr)

    # Take into account possible divisions by zero
    precision = np.true_divide(diag, predicted_sum, np.full(diag.shape, np.nan), where=predicted_sum != 0)
    recall = np.true_divide(diag, gt_sum, np.full(diag.shape, np.nan), where=gt_sum != 0)
    num = 2 * (precision * recall)
    den = (precision + recall)
    f1 = np.true_divide(num, den, np.full(num.shape, np.nan), where=den!=0)
    return precision, recall, f1, accuracy

def train(model, criterion: nn.Module, optimizer, train_loader, test_loader, epochs, squeeze, scheduler=None, n_loss_print=1, device=None, classification: bool=True, wb=False):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
    #     print('%d gpus detected' % torch.cuda.device_count())
    #     model = nn.DataParallel(model)
    #     model = model.to(device)

    mytype = torch.long if classification else torch.float
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        for idx, data in enumerate(train_loader):
            image, mask = data['image'], data['mask']

            image = image.to(device)
            mask = mask.to(device, dtype=mytype)
            if squeeze:
                mask = mask.squeeze(dim=1)

            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            
            iou = binary_mean_iou(output, mask)
            running_loss += loss.item()
            epoch_loss += loss.item()

            if idx % n_loss_print == (n_loss_print - 1):
                r_loss = running_loss / n_loss_print
                print('[%d - %5d] loss: %.3f' % (epoch + 1, idx + 1, r_loss))
                if wb: 
                    wandb.log({"loss": r_loss})
                    wandb.log({"train_iou": iou})
                running_loss = 0
                
        running_loss = 0
        
        validate(model, criterion, test_loader, device=device, squeeze=squeeze, classification=classification)
        if scheduler is not None:
            scheduler.step()
        
        with torch.no_grad():
            model.eval()
            for idx, data in enumerate(test_loader):
                image, mask = data['image'], data['mask']

                image = image.to(device)
                mask = mask.to(device, dtype=mytype)
                if squeeze:
                    mask = mask.squeeze(dim=1)

                output = model(image)
                loss = criterion(output, mask)

                iou = binary_mean_iou(output, mask)
                running_loss += loss.item()
                epoch_loss += loss.item()

                if idx % n_loss_print == (n_loss_print - 1):
                    r_loss = running_loss / n_loss_print
                    print('[%d - %5d] loss: %.3f' % (epoch + 1, idx + 1, r_loss))
                    if wb: 
                        wandb.log({"val_loss": r_loss})
                        wandb.log({"val_iou": iou})
                    running_loss = 0

    return

def train_concatenated_model(model, criterion: list, optimizer, train_loader, test_loader, epochs, scheduler=None, n_loss_print=1, device=None, one_hot=False, threshold: float=0.5):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
    #     print('%d gpus detected' % torch.cuda.device_count())
    #     model = nn.DataParallel(model)
    #     model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        for idx, data in enumerate(train_loader):
            image, mask = data['image'], data['mask']

            image = image.to(device)
            mask = mask.to(device)
            if not one_hot:
                bin_mask = torch.where(mask >= 1.0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            else:
                bin_mask = torch.where(mask.argmax(dim=1) >= 1.0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

            optimizer.zero_grad()
            bin_out, regr_out = model(image)

            loss = 0.0
            loss += criterion[0](bin_out, bin_mask)
            loss += criterion[1](regr_out, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if idx % n_loss_print == (n_loss_print - 1):
                print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss))
                running_loss = 0.0

        validate_concatenated_model(model, criterion, test_loader, device, one_hot, threshold=threshold)
        if scheduler is not None:
            scheduler.step()

    return

def validate(model, criterion: nn.Module, loader, squeeze: bool, device=None, classification: bool=True):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
    #     print('%d gpus detected' % torch.cuda.device_count())
    #     model = nn.DataParallel(model)
    #     model = model.to(device)

    model.eval()
    tot_sum = 0
    correct = 0
    always_0 = 0
    mytype = torch.long if classification else torch.float
    with torch.no_grad():
        for idx, data in enumerate(loader):
            image, mask = data['image'], data['mask']

            image = image.to(device)
            mask = mask.to(device, dtype=mytype)
            if mask.shape[1] > 1:
                mask = mask.argmax(dim=1, keepdim=True)
            if squeeze:
                mask = mask.squeeze(dim=1)

            output = model(image)
            loss = criterion(output, mask)

            if classification:
                output = output.argmax(dim=1, keepdim=(not squeeze))
            else:
                output = output.round()

            correct += (mask == output).sum().item()
            tot_sum += mask.numel()
            always_0 += (mask == 0).sum().item()
            
    print('Accuracy: %f' % (correct / tot_sum))
    print('Always-0 predictor performance: %f' % (always_0 / tot_sum))
    return


def binary_mean_iou2(logits, targets, EPSILON=1e-6) -> torch.Tensor:
    output = (logits > 0.5).int()
    targets = (targets > 0.5).int()
    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result


def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()


def binary_mean_iou(outputs: torch.Tensor, labels: torch.Tensor, SMOOTH=1e-6, average=True):
    if outputs.shape[1] == 1:
        outputs = outputs.squeeze(1)
    if labels.shape[1] == 1:
        labels = labels.squeeze(1)
        
    outputs = (outputs > .5)
    labels = (labels > .5)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
#     intersection = (outputs & labels).float().sum()
#     union = (outputs | labels).float().sum()
    if average:
        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
        iou = iou.mean()
        return iou
    
    return intersection, union


def class_IU(gt, prediction, n_classes=5, get_tensor=False):
    """Takes as input gt and prediction, returns intersection and union as the sum of intersections and unions
    of each of the n_classes classes.
    Intersection and Union are taken only for classes != 0
    """
    if not gt.shape == prediction.shape:
        raise RuntimeError(f'Multiclass IoU: size mismatch. ({gt.shape} and {prediction.shape})')

    intersection, union = 0, 0
    for current_class in range(1, n_classes):
        pred_inds = (prediction == current_class)
        gt_inds = (gt == current_class)

        if gt_inds.long().sum().item() == 0:
            continue
        else:
            intersection_now = (pred_inds[gt_inds]).long().sum().item()
            intersection += intersection_now
            union += (pred_inds.long().sum().item() + gt_inds.long().sum().item() - intersection_now)
    
    if get_tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(intersection, device=device), torch.tensor(union, device=device)

    return intersection, union

def class_IU_from_cm(cm):
    n_class = cm.shape[0]

    I = cm.diagonal()
    U = np.array([cm[c,:].sum()+cm[:,c].sum()-cm[c, c] for c in range(n_class)])

    #TO DO: eventually exclude class 0
    return I, U

def validate_concatenated_model(model, criterion: list, loader, device=None, one_hot: bool=False, threshold: float=0.5, intersection_over_union: bool=True):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
    #     print('%d gpus detected' % torch.cuda.device_count())
    #     model = nn.DataParallel(model)
    #     model = model.to(device)

    model.eval()
    tot_sum = 0
    correct_bin = 0
    correct_regr = 0
    intersection = 0
    union = 0
    always_0 = 0
    sigm = nn.Sigmoid()

    with torch.no_grad():
        for idx, data in enumerate(loader):
            image, mask = data['image'], data['mask']

            image = image.to(device)
            mask = mask.to(device)
            if not one_hot:
                bin_mask = torch.where(mask >= 1.0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            else:
                bin_mask = torch.where(mask.argmax(dim=1) >= 1.0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

            bin_out, regr_out = model(image)
            bin_out_pred = torch.where(sigm(bin_out) > threshold, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
            regr_out_pred = regr_out.round()
            
            loss = 0.0
            loss += criterion[0](bin_out, bin_mask)
            loss += criterion[1](regr_out, mask)

            # Each intersection/ union contributes to the summation
            outputs = bin_out_pred.byte()
            labels = bin_mask.byte()

            intersection += (outputs & labels).float().sum()
            union += (outputs | labels).float().sum()

            correct_bin += (bin_mask == bin_out_pred).sum().item()
            correct_regr += (mask == regr_out_pred).sum().item()
            always_0 += (mask == 0).sum().item()
            tot_sum += mask.numel()

    eps = 1e-6
    print('Binary UNet Intersection over Union: %.3f' % ((intersection + eps) / (union + eps))) # smoothed division to avoid 0/0
    print('Binary UNet accuracy: %.3f' % (correct_bin / tot_sum))
    print('Regression rounded accuracy: %.3f' % (correct_regr / tot_sum))
    print('Always-0 model accuracy: %.3f' % (always_0 / tot_sum))
                
    return

def initialize_weight(model, seed=None):
    """
    Random initialization of network weights
    """
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d) or isinstance(model, nn.Linear):
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.xavier_normal_(model.weight.data)
        if seed is not None:
            torch.manual_seed(seed)
        nn.init.normal_(model.bias.data)
    elif type(model) is PSPNet:
        model.initialize_weights(seed)