import os
import argparse
from neural_net.utils import str2bool

def get_args():
    parser = argparse.ArgumentParser(description='Training of the U-Net usign Pytorch Lightining framework.')
    parser.add_argument('--discard_images', nargs='?', default=False, const=True, help = "Prevent Wandb to save validation result for each step.")
    parser.add_argument('-k', '--key', type=str, default='purple', help = "Test set fold key. Default is 'blue'.")
    parser.add_argument('-m', '--model_name', type=str, default='unet', help = "Select the model (unet, canet or attnet available). Default is unet.")
    parser.add_argument('--losses', type=str, default=None, help = "Select the configuration name of the loss function(s). The name must be written in lower casses without special characters.")
    
    parser.add_argument('--lr', type=float, default=None, help = "Custom lr.")
    parser.add_argument('--batch_size', type=int, default=None, help = "Custom batch size.")
    parser.add_argument('--seed', type=int, default=7, help = "Custom seed.")
#     parser.add_argument('--encoder', type=str, default='resnet34', help = "Select the model encoder (only available for smp models). Default is resnet34.")
    
    args = parser.parse_args()

    return args

import ast
from pathlib import Path
import pickle
import wandb
from easydict import EasyDict as ed

import torch
from neural_net.cnn_configurations import TrainingConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from base_train import Satmodel
import neural_net
import matplotlib.pyplot as plt


def train(args):        
    hparams = TrainingConfig(**vars(args))
#                             , n_channels=24, mode='both')
#                             , only_burnt=False)
    
    if not torch.cuda.is_available():
        hparams.trainer.gpus = 0
        hparams.trainer.precision = 32

    name = f'test_{args.model_name}_{args.key}'.lower()
    if args.losses: name += f"_{args.losses.lower()}"
    if args.seed is not None: name += f"_{args.seed}"
    
    outdir = Path("../data/new_ds_logs/Propaper")
    outdir = outdir / name
    outdir.mkdir(parents=True, exist_ok=True)
    
    if any(outdir.glob("*best*")):
        print(f"Simulation already done ({name})")
        return
    
    run = wandb.init(reinit=True, project="rescue_paper", entity="smonaco", name=name, settings=wandb.Settings(start_method='fork'))
    
    print(f'Best checkpoints saved in "{outdir}"')

    pl_model = Satmodel(hparams, {'log_imgs': not args.discard_images})
    
    earlystopping_callback = EarlyStopping(**hparams.earlystopping)
    hparams["checkpoint"]["dirpath"] = outdir
    checkpoint_callback = ModelCheckpoint(**hparams.checkpoint)
    
    logger = WandbLogger(save_dir=outdir, name=name)
    logger.log_hyperparams(hparams)
    logger.watch(pl_model, log='all', log_freq=1)
    
    trainer = pl.Trainer(
        **hparams.trainer,
        max_epochs=hparams.epochs,
#         auto_scale_batch_size='binsearch',
        logger=logger,
        callbacks=[checkpoint_callback,
                   earlystopping_callback
                   ],
    )
    get_lr = False
    
    trainer.tune(pl_model)
    
    if get_lr:
        lr_finder = trainer.tuner.lr_find(pl_model, 
                            min_lr=0.00005, 
                            max_lr=0.001,
                            mode='linear')

        # Plots the optimal learning rate
        fig = lr_finder.plot(suggest=True)

        fig.imsave('best_lr.png')
        wandb.log({"best_lr": fig})
        hparams.optimizer.lr = lr_finder.suggestion()

    trainer.fit(pl_model)
    
    best = Path(checkpoint_callback.best_model_path)
    best.rename(best.parent / f'{wandb.run.name}-best{best.suffix}')
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
    