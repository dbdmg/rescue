import os
import torch

from torch import nn

class EarlyStopping():
    def __init__(self, patience, save_path, tol: float, verbose: bool=False, save_best: bool=True):
        self.patience = patience
        self.save_path = save_path
        self.tol = tol
        self.verbose = verbose
        self.save_best = save_best

        assert os.path.isdir(save_path)

        self.save_path = os.path.join(save_path, 'checkpoint.pt')
        self.epoch_counter = 0
        self.loss_min = None
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.loss_min is None or val_loss < self.loss_min - self.tol:
            self.loss_min = val_loss
            self.epoch_counter = 0

            if not self.save_best:
                if self.verbose:
                    print('    ---- EarlyStopper - counter reset. Saving model (save_best is set to False) ----')
                    bst = self.best_loss if self.best_loss is not None else 0.0
                    print(f'    ---- Best value was {bst}, new value is {val_loss} ----')
                self.best_loss = val_loss
                torch.save(model.state_dict(), self.save_path)
        else:
            self.epoch_counter += 1
            if self.verbose:
                print(f'    ---- EarlyStopper - counter = {self.epoch_counter}, patience = {self.patience} ----')

            if self.epoch_counter >= self.patience:
                if self.verbose:
                    print(f'    ---- EarlyStopper - patience = {self.patience} reached. ----')
                self.early_stop = True

        if (self.best_loss is None or val_loss < self.best_loss) and self.save_best:
            if self.verbose:
                bst = self.best_loss if self.best_loss is not None else 0.0
                if bst != 0:
                    print(f'    ---- EarlyStopper - saving model. Old loss = {bst}, new loss = {val_loss} ----')
                else:
                    print(f'    ---- EarlyStopper - saving model. New loss = {val_loss} ----')
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.save_path)