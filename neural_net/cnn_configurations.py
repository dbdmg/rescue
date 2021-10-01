import os
import yaml
import re
from pathlib import Path
from easydict import EasyDict as ed

from neural_net.cross_validator import ConcatenatedCrossValidator, CrossValidator, GradcamCrossValidator
from neural_net import *
from neural_net.transform import *
from neural_net.loss import IndexLoss, IoULoss, FuzzyIoULoss, GDiceLossV2, ComboLoss, softIoULoss, F1MSE
from neural_net.performance_storage import AccuracyBinStorage, AccuracyAllStorage, AccuracySingleRegrStorage

from neural_net.unet import ConcatenatedUNet, UNet, MixedNet
from neural_net.pspnet import PSPNet
from neural_net.nested_unet import NestedUNet, ConcatenatedNestedUNet
from neural_net.segnet import SegNet, ConcatenatedSegNet

from torch import nn, optim
import pandas as pd

validation_dict = {'purple': 'coral',
                   'coral': 'cyan',
                   'pink': 'coral',
                   'grey': 'coral',
                   'cyan': 'coral',
                   'lime': 'coral',
                   'magenta': 'coral'
                  }

mask_intervals = [(0, 36), (37, 96), (97, 160), (161, 224), (225, 255)]
#     {
#         "blue": "fucsia",
#         "brown": "fucsia",
#         "fucsia": "green",
#         "green": "fucsia",
#         "orange": "fucsia",
#         "red": "fucsia",
#         "yellow": "fucsia",
#     }


def TrainingConfig(**args):
    with open(Path.cwd() / "configs/models.yaml", "r") as f:
        models = ed(yaml.load(f, Loader=yaml.SafeLoader))
        
    with open(Path.cwd() / "configs/losses.yaml", "r") as f:
        losses = ed(yaml.load(f, Loader=yaml.SafeLoader))
    
    mod_name = next(f for f in models.keys() if args["model_name"].lower() == f.lower())
    
#     if (Path.cwd() / f"configs/{mod_name}.yaml").exists():
#         with open(Path.cwd() / f"configs/{mod_name}.yaml", "r") as f:
#             hparams = ed(yaml.load(f, Loader=yaml.SafeLoader))
#     else:
    with open(Path.cwd() / "configs/UNet.yaml", "r") as f:
        hparams = ed(yaml.load(f, Loader=yaml.SafeLoader))
        hparams.model = models[mod_name]
            
    if args["losses"]:
        loss_key = next((key for key in losses.config_names.keys() if args["losses"] in re.sub('[^A-Za-z0-9]+', '', key).lower()))
        hparams.criterion = losses.classification[losses.config_names[loss_key].first]
        hparams.regr_criterion = losses.regression[losses.config_names[loss_key].second]
        
    for k in args:
        if args[k] is None or str(k) in ['model_name', 'discard_images', 'encoder', 'losses']: continue
        found = False
        if k in hparams.keys():
            hparams[k] = args[k]
            found = True
        for k_n in hparams.keys():
            if type(hparams[k_n]) in [dict, ed] and k in hparams[k_n].keys():
                hparams[k_n][k] = args[k]
                found = True
        if not found: print(f"\nParameter `{k}` not found.\n")
    
    print(f"Selecting model {mod_name} as backbone")
    
    hparams.dataset_specs.mask_intervals = mask_intervals
    
    hparams.groups = read_groups(hparams.fold_separation_csv)
    
    hparams.validation_dict = validation_dict
    
    if "imagenet" in hparams.model.values() and False:
        print("\n> Using imagenet preprocessing.")
        mn = [1 for i in range(hparams.model.n_channels)]
        std = [1 for i in range(hparams.model.n_channels)]
        mn[1:4] = (0.406, 0.456, 0.485)  # rgb are 3,2,1
        std[1:4] = (0.225, 0.224, 0.229)
    else:
        mn = (0.5,) * hparams.model.n_channels
        std = (0.5,) * hparams.model.n_channels
            
    # Dataset augmentation and normalization
    hparams.train_transform = transforms.Compose([
        RandomRotate(0.5, 50, seed=hparams.seed),
        RandomVerticalFlip(0.5, seed=hparams.seed),
        RandomHorizontalFlip(0.5, seed=hparams.seed),
        RandomShear(0.5, 20, seed=hparams.seed),
        ToTensor(round_mask=True),
#             Resize(800),
        Normalize(mn, std)
    ])
    hparams.test_transform = transforms.Compose([
        ToTensor(round_mask=True),
#             Resize(800),
        Normalize(mn, std)
    ])
    
    return hparams
    
def update_transforms():
    print("\nUsing pretraining for imagenet weights.\n")

    imgnet_mean = [1 for i in range(self.n_channels)]
    imgnet_std = [1 for i in range(self.n_channels)]
    imgnet_mean[1:4] = (0.406, 0.456, 0.485)  # rgb are 3,2,1
    imgnet_std[1:4] = (0.225, 0.224, 0.229)
    mn = imgnet_mean
    std = imgnet_std

    self.train_transform = transforms.Compose([
        RandomRotate(0.5, 50, seed=self.seed),
        RandomVerticalFlip(0.5, seed=self.seed),
        RandomHorizontalFlip(0.5, seed=self.seed),
        RandomShear(0.5, 20, seed=self.seed),
        ToTensor(round_mask=True),
        Resize(800),
        Normalize(imgnet_mean, imgnet_std)
    ])
    self.test_transform = transforms.Compose([
        ToTensor(round_mask=True),
        Resize(800),
        Normalize(imgnet_mean, imgnet_std)
    ])
        
def read_groups(satellite_folds, verbose=False):
    """
    Read folds (i.e., colors) - for each fold get the corresponding input folders of Sentinel-2 dataset
    @return dictionary: key = fold color, value = list of dataset folders in this fold
    """
    groups = {}
    df = pd.read_csv(satellite_folds)
    for key, grp in df.groupby('fold'):
        folder_list = grp['folder'].tolist()

        if verbose==True:
            print('______________________________________')
            print(f'fold key: {key}')
            print(f'folders ({len(folder_list)}): {str(folder_list)}')
        groups[key] = folder_list
    return groups
