import os
import ast
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Union, Dict, List, Tuple

from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from neural_net import *
from neural_net.sampler import ShuffleSampler
from neural_net.loss import GDiceLossV2
from neural_net.utils import *
from neural_net.transform import *
from neural_net.transf_learning import *

import segmentation_models_pytorch as smp
from neural_net.unet import UNet
from neural_net.attention_unet import AttentionUnet
from neural_net.canet_parts.networks.network import Comprehensive_Atten_Unet


import pytorch_lightning as pl

import wandb

def c_binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor, EPSILON = 1e-15) -> torch.Tensor:
    
    output = (logits > 0.5).int()
    intersection = (targets * output).sum()
    union = targets.sum() + output.sum() - intersection
    result = (intersection + EPSILON) / (union + EPSILON)

    return result
    
class Satmodel(pl.LightningModule):
    def __init__(self, hparams, opt):
        super().__init__()

        for key in ["binary", "log_imgs", "log_res"]:
            setattr(self, key, opt.get(key, True))
            
        self.hparams.update(hparams)
        
        if not "SLURM_JOB_ID" in os.environ:
            hparams.num_workers = 0
            hparams.batch_size = 8
        
        self.save_hyperparameters()
        self.model = self.get_model()
        self.drop_last = False
        
        if self.binary:
            self.criterion = eval_object(hparams.criterion)
        else:
            self.criterion = eval_object(hparams.regr_criterion)

        self.train_set = []
        self.validation_set = []
        self.test_set = []
        self.add_nbr = False

    def forward(self, batch):
        return self.model(batch)
    
    def get_model(self):
        model = eval_object(self.hparams.model)
        if type(model) == UNet:
            model.apply(initialize_weight)

    #         model = Comprehensive_Atten_Unet(in_ch=12, n_classes=1, im_size=(480, 480))
        return model
    

    def configure_optimizers(self):
        optimizer = eval_object(self.hparams.optimizer, params=self.model.parameters())
        self.optimizers = [optimizer]
        
        if self.hparams.get('scheduler'):
            scheduler = eval_object(self.hparams.scheduler, optimizer)
            
            if self.hparams.scheduler.name == optim.lr_scheduler.ReduceLROnPlateau:
                return {
                   'optimizer': optimizer,
                   'lr_scheduler': scheduler,
                   'monitor': 'val_loss'
               }
            return self.optimizers, [scheduler]
        return self.optimizers

    def setup(self, stage=0):
        self.hparams.num_workers = 0
        self.hparams.batch_size = 8
        ordered_keys = list(self.hparams.groups.keys())

        validation_fold_name = self.hparams.validation_dict[self.hparams.key]
        self.validation_set = self.hparams.groups[validation_fold_name]
        print(f'Test set is {self.hparams.key}, validation set is {validation_fold_name}. All the rest is training set.')

        for grp in self.hparams.groups:
            if grp == validation_fold_name or grp == self.hparams.key:
                continue
            else:
                self.train_set.extend(self.hparams.groups[grp])

        self.test_set = self.hparams.groups[self.hparams.key]
#         if self.binary:
#             print(f'Training set ({len(self.train_set)}): {str(self.train_set)}')
#             print(f'Validation set ({len(self.validation_set)}): {str(self.validation_set)}')

    def train_dataloader(self) -> DataLoader:
        train_dataset = SatelliteDataset(folder_list=self.train_set,
                                         transform=self.hparams.train_transform,
                                         **self.hparams.dataset_specs)
        
        train_sampler = ShuffleSampler(train_dataset, self.hparams.seed)

        result = DataLoader(train_dataset, 
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            sampler=train_sampler,
                            pin_memory=True,
                            drop_last=self.drop_last
                           )
        print(f'Train set dim: {len(train_dataset)} ({len(result)} batches)')
                
        return result

    def val_dataloader(self):
        validation_dataset = SatelliteDataset(folder_list=self.validation_set,
                                              transform=self.hparams.test_transform,
                                              **self.hparams.dataset_specs)
                
        validation_sampler = ShuffleSampler(validation_dataset, self.hparams.seed)

        result = DataLoader(validation_dataset, 
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            sampler=validation_sampler,
                            pin_memory=True,
                            drop_last=self.drop_last
                           )
        print(f'Validation set dim: {len(validation_dataset)} ({len(result)} batches)')
                
        return result
    
    def test_dataloader(self):
        test_dataset = SatelliteDataset(folder_list=self.test_set,
                                              transform=self.hparams.test_transform,
                                              **self.hparams.dataset_specs)
                
        result = DataLoader(test_dataset, 
                            batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=True,
                            drop_last=False
                           )   
        print(f'Validation set dim: {len(test_dataset)} ({len(result)} batches)')
        return result

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]
        
        if self.binary: masks = (masks > .5).type_as(masks)
        logits = self.forward(images)
        loss = self.criterion(logits, masks)
        
        if wandb.run is not None:
            if self.binary:
                self.log('train_iou', binary_mean_iou(logits, masks))
            else:
                sq_err, counters = compute_squared_errors(logits, masks, len(self.hparams.dataset_specs.mask_intervals))
                mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)
                self.log('train_rmse', np.sqrt(mse[-1]))

            self.log('lr', self._get_current_lr())
            self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        logits = self.forward(images)
        if self.binary: masks = (masks > .5).type_as(masks)

        loss = self.criterion(logits, masks)
        if self.log_res: self.log("val_loss", loss)
        
        if self.binary:
            val_kpi = binary_mean_iou(logits, masks)
            if wandb.run is not None:
                self.log('val_iou', val_kpi)
        else:
            sq_err, counters = compute_squared_errors(logits, masks, len(self.hparams.dataset_specs.mask_intervals))
            mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)
            val_kpi = np.sqrt(mse[-1])
            
            if wandb.run is not None:
                if self.log_res: self.log('val_rmse', val_kpi)
            
        if self.log_imgs and self.log_res:
            self.log_images(images, logits, masks)

        return {'val_loss': loss}
    
    def log_images(self, images, logits, masks, log_dist=3):
        if self.binary:
            class_labels = {0: "background", 1: "fire"}

            logits_ = (torch.sigmoid(logits) > 0.5).cpu().detach().numpy().astype("float")
            masks_ = (masks > 0.5).cpu().detach().numpy().astype("float")
        else:
            class_labels = {0: "unburnt", 1: "1", 2: "2", 3: "3", 4: "4"}

            logits_ = logits.cpu().detach().numpy().astype("float")
            masks_ = masks.cpu().detach().numpy().astype("float")

        if self.trainer.current_epoch % log_dist == 0 and wandb.run is not None:
            for i in range(images.shape[0]):
                mask_img = wandb.Image(
                    images[i, [3,2,1], :, :]*2.5,
                    masks={
                        "predictions": {
                            "mask_data": logits_[i, 0, :, :],
                            "class_labels": class_labels,
                        },
                        "groud_truth": {
                            "mask_data": masks_[i, 0, :, :],
                            "class_labels": class_labels,
                        },
                    },
                )
                self.logger.experiment.log({"val_images": [mask_img]}, commit=False)

    def _get_current_lr(self):
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]  # type: ignore

        if torch.cuda.is_available(): return torch.Tensor([lr])[0].cuda()
        return torch.Tensor([lr])[0]
    
    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss}

#     def validation_epoch_end(self, outputs):
#         self.log("epoch", self.trainer.current_epoch)

#         if self.binary:
#             avg_val_iou = find_average(outputs, "val_iou")
#             self.log("val_iou", avg_val_iou)
#         else:
#             avg_val_rmse = find_average(outputs, "val_rmse")
#             self.log("val_rmse", avg_val_rmse)
#         return

class Double_Satmodel(Satmodel):
    def __init__(self, hparams, opt):
        super().__init__(hparams, opt)
        
        if self.binary:
            print(f'\n    Iteration step 1/2 - Binary network training...  (test on {self.hparams.key} fold)\n')
        else:
            print(f'\n    Iteration step 2/2 - Regression network training...  (test on {self.hparams.key} fold)\n')
            
        self.set_model()
    
    def set_model(self):
        self.hparams.backbone = self.model.backbone
        if self.binary:
            if type(self.model.binary_unet) == UNet: self.model.apply(initialize_weight)
            self.model.unfreeze_binary_unet()
            self.model.freeze_regression_unet()
        else:
            self.model.freeze_binary_unet()
            self.model.unfreeze_regression_unet()
    
    def forward(self, batch):
        if self.binary:
            self.model.regression_unet.eval()
            self.model.freeze_regression_unet()
            return self.model(batch)[0]
        else:
            self.model.binary_unet.eval()
            self.model.freeze_binary_unet()
            return self.model(batch)[1]
#             mytype = torch.float32

    def configure_optimizers(self):
        if self.binary:
            optimizer = eval_object(self.hparams.optimizer, params=self.model.binary_unet.parameters())
        else:
            optimizer = eval_object(self.hparams.optimizer, params=self.model.parameters())
        self.optimizers = [optimizer]
        
        if self.hparams.get('scheduler'):
            scheduler = eval(self.hparams.scheduler.pop('name'))(optimizer, **self.hparams.scheduler)
            
            if self.hparams.scheduler.name == optim.lr_scheduler.ReduceLROnPlateau:
                return {
                   'optimizer': optimizer,
                   'lr_scheduler': scheduler,
                   'monitor': 'val_loss'
               }
            return self.optimizers, [scheduler]
        return self.optimizers
    
    def test_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        bin_pred, regr_pred = self.model(images)
        
        bin_pred = torch.sigmoid(bin_pred).squeeze().cpu().detach().numpy() > 0.5
        
        intersection, union = binary_mean_iou(masks.cpu().detach().numpy(), bin_pred, average=False)
        
        intersection = intersection.sum()
        union = union.sum()
        
        regr_pred = regr_pred.clamp(0, max=(len(self.hparams.dataset_specs.mask_intervals) - 1))
        tmp_sq_err, tmp_counters = compute_squared_errors(regr_pred, masks,
                                                          len(self.hparams.dataset_specs.mask_intervals)
                                                         )
        
        severity_pred = regr_pred.round().squeeze().cpu().detach().numpy().astype("float")
        
        severity_pred *= (255.0/(len(self.hparams.dataset_specs.mask_intervals) - 1))
        
        out_path = self.hparams.checkpoint.dirpath / "predictions"
        out_path.mkdir(parents=True, exist_ok=True)
        for i in range(images.shape[0]):
            Image.fromarray(255*bin_pred[i].astype(float)).convert('RGB').save(out_path / f"bin_{batch_idx*self.hparams.batch_size + i}.png")
            Image.fromarray(severity_pred[i]).convert('RGB').save(out_path / f"sev_{batch_idx*self.hparams.batch_size + i}.png")
            
        return {'regr': [tmp_sq_err, tmp_counters], 'bin': [intersection, union]}
    
    def test_epoch_end(self, outputs):
        r_outputs = np.array([out['regr'] for out in outputs])
        bin_outputs = np.array([out['bin'] for out in outputs])
                
        sqe = r_outputs[:, 0, :].sum(axis=0)
        counters = r_outputs[:, 1, :].sum(axis=0)
        
        intersections, unions = bin_outputs.sum(0)
        
        ious = (intersections + 1e-6) / (unions + 1e-6)
        pd.DataFrame([intersections, unions, ious],
                     index=['intersection', 'union', 'iou'],
                     columns=['value'],
                    ).to_csv(self.hparams.checkpoint.dirpath / "bin_kpi.csv")
        
        mse = np.true_divide(sqe, counters, np.full(sqe.shape, np.nan), where=counters != 0)
        pd.DataFrame([sqe, counters, np.sqrt(mse)],
                     index=['sqe', 'count', 'rmse'],
                    ).rename(columns={5: 'all'}).to_csv(self.hparams.checkpoint.dirpath / "regr_kpi.csv")
    
