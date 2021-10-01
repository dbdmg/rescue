import os
import argparse
import ast
from pathlib import Path
import pickle
import wandb
from easydict import EasyDict as ed

import torch
from neural_net.cnn_configurations import TrainingConfig
from neural_net.utils import str2bool

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from base_train import Satmodel, Double_Satmodel
import neural_net
from run_single import get_args


def train(args):

    
    hparams = TrainingConfig(**vars(args))
#                             , n_channels=24, mode='both')
#                             , only_burnt=False)
        
    if "SLURM_JOB_ID" in os.environ:
        print("Running in Slurm")
        hparams["job_id"] = os.environ["SLURM_JOB_ID"]
        hparams.num_workers = 4
        hparams.batch_size = 16
        
    hparams.model = {
        "name": "ConcatenatedModel",
        "model_dict": hparams.model
    }
    name = f'test_double-{args.model_name}_{args.key}'.lower()
    if args.losses: name += f"_{args.losses.lower()}"
    if args.seed is not None: name += f"_{args.seed}"
    
    outdir = Path(f"../data/new_ds_logs/Propaper/legion/{name}")#_imgnotnorm")
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    if (outdir / 'bin_kpi.csv').exists() and (outdir / 'regr_kpi.csv').exists():
        print(f"Simulation already done ({name})")
        return
        
    run = wandb.init(reinit=True, project="rescue_paper", entity="smonaco", name=name, tags=["crossval_double"], settings=wandb.Settings(start_method='fork'))
    
    print(f'Best checkpoints saved in "{outdir}"\n')

    hparams.checkpoint.dirpath = outdir
    
    if not torch.cuda.is_available():
        hparams.trainer.gpus = 0
        hparams.trainer.precision = 32
        
    logger = WandbLogger(save_dir=outdir, name=name)
    logger.log_hyperparams(hparams)
#     logger = None
    
    #### 1st network ###################
    if not any(outdir.glob("bin*best*")):
        earlystopping_1 = EarlyStopping(**hparams.earlystopping)
        checkpoint_1 = ModelCheckpoint(**hparams.checkpoint, filename='binary_model-{epoch}')
        
        bin_model = Double_Satmodel(hparams, {'log_imgs': not args.discard_images, 'binary': True, 'log_res':True})

        logger.watch(bin_model, log='all', log_freq=1)
        trainer = pl.Trainer(
            **hparams.trainer,
            max_epochs=hparams.epochs,
            logger=logger,
            callbacks=[checkpoint_1,
                       earlystopping_1
                       ],
        )
        trainer.fit(bin_model)
        best_path = Path(checkpoint_1.best_model_path)
        best = str(best_path.parent / f'{best_path.stem}_best{best_path.suffix}')
        best_path.rename(best)
    else:
        print("> Resuming from intermediate step.")
        best = str(next(outdir.glob("bin*best*")))

    #### 2nd network ###################
    if not any(outdir.glob("reg*best*")):
        earlystopping_2 = EarlyStopping(**hparams.earlystopping)
        checkpoint_2 = ModelCheckpoint(**hparams.checkpoint, filename='regression_model-{epoch}')
        
#         intermediate_chp = next(outdir.glob("reg*.ckpt"), None)
#         if intermediate_chp:
#             best = str(intermediate_chp)
            
        regr_model = Double_Satmodel.load_from_checkpoint(best,#checkpoint_1.best_model_path,
                                                          opt={'log_imgs': not args.discard_images, 'binary': False, 'log_res':True}
                                                  )
        trainer = pl.Trainer(
            **hparams.trainer,
            max_epochs=hparams.epochs,
            logger=logger,
            callbacks=[checkpoint_2,
                       earlystopping_2
                       ],
        )
        trainer.fit(regr_model)
        
        trainer.test()
        best_path = Path(checkpoint_2.best_model_path)
        best_path.rename(best_path.parent / f'{best_path.stem}_best{best_path.suffix}')
        checkpoint_2.best_model_path = str(best_path)
    else:
        print("> Resuming weights for evaluation.")
        regr_model = Double_Satmodel.load_from_checkpoint(str(next(outdir.glob("reg*best*"))), hparams=hparams)
        trainer = pl.Trainer(**hparams.trainer, logger=logger).test(regr_model)
    
    wandb.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
    