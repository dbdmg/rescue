import torch
from PIL import Image
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from .dataset import SatelliteDataset
from .image_processor import ProductProcessor
from .transform import *
from .index_functions import *
from .sampler import ShuffleSampler
from .stopping import EarlyStopping
from .utils import compute_prec_recall_f1_acc, compute_squared_errors, initialize_weight, class_IU, class_IU_from_cm
import cv2

from pickle import dump

from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score

from collections import OrderedDict, defaultdict

from neural_net.unet import UNet

def print_delta_time(delta):
    """Format delta time:"""
    # delta is in seconds
    h = delta // 3600
    h_rem = delta % 3600
    m = h_rem // 60
    s = h_rem % 60
    return f"{h}h:{m}m:{s}s"

def predict_remaining_time(avgdelta, current_done, total):
    return (total-current_done)*avgdelta

class CrossValidator():
    def __init__(self, groups, model_tuple: tuple, criterion_tuple: tuple, train_transforms, test_transforms, master_folder: str, csv_path: str, epochs: int, batch_size: int, lr: float, wd: float, product_list: list, mode, process_dict: dict, mask_intervals: list, mask_one_hot: bool, height: int, width: int, filter_validity_mask: bool, only_burnt: bool, mask_filtering: bool, seed: int, result_folder: str, lr_scheduler_tuple: tuple=None, ignore_list: list=None, performance_eval_func=None, squeeze_mask=True, early_stop=False, patience=None, tol=None, is_regression=False, validation_dict=None):
        self.groups = groups
        if isinstance(groups, list):
            self.groups = self._convert_list_to_dict(groups)
        assert len(model_tuple) == 2
        assert len(criterion_tuple) == 2
        assert lr_scheduler_tuple is None or len(lr_scheduler_tuple) == 2
        self.model_tuple = model_tuple
        self.criterion_tuple = criterion_tuple
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.master_folder = master_folder
        self.csv_path = csv_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.product_list = product_list
        self.mode = mode
        self.process_dict = process_dict
        self.mask_intervals = mask_intervals
        self.mask_one_hot = mask_one_hot
        self.height = height
        self.width = width
        self.filter_validity_mask = filter_validity_mask
        self.only_burnt = only_burnt
        self.mask_filtering = mask_filtering
        self.seed = seed
        self.result_folder = result_folder
        self.ignore_list = ignore_list
        self.lr_scheduler_tuple = lr_scheduler_tuple
        self.performance_eval_func = performance_eval_func
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.squeeze_mask = squeeze_mask
        self.early_stop = early_stop
        self.patience = patience
        self.tol = tol
        self.is_regression = is_regression
        self.validation_dict = validation_dict

        assert set(validation_dict.keys()) == set(groups.keys())

        if self.early_stop:
            assert patience is not None and tol is not None
        
        self._create_folder(self.result_folder)
        return

    @classmethod
    def _create_folder(cls, path):
        if not os.path.isdir(path):
            os.mkdir(path)
            if not os.path.isdir(path):
                raise RuntimeError('Unable to generate folder at %s' % path)

        return

    @classmethod
    def _convert_list_to_dict(cls, group_list):
        assert isinstance(group_list, list)

        result = OrderedDict()
        for idx, sublist in enumerate(group_list):
            result[idx] = sublist

        return result

    @classmethod
    def _instantiate(cls, item_tuple):
        if item_tuple is None:
            return None
        assert len(item_tuple) == 2 and isinstance(item_tuple[1], dict)
        item_class, item_args = item_tuple[0], item_tuple[1]
        return item_class(**item_args)

    def start(self, mask_postfix='mask', compute_cm=True):
        print('Running cross validation...')
        df_dict = defaultdict(list)

        ordered_keys = list(self.groups.keys())

        fold_times = []
        for idx, key in enumerate(ordered_keys):
            print('\n_______________ CROSS-VALIDATION ITERATION _______________')
            if len(fold_times)>1:
                avg_sec = sum(fold_times)/len(fold_times)
                remain = predict_remaining_time(avg_sec, len(fold_times), len(ordered_keys))
                print(f"Average fold processing time: {print_delta_time(avg_sec)}, remaining time: {print_delta_time(remain)}")
            start_time = time.time() # Measuring epoch time
            print(f'Testing on {key} fold ({idx + 1} of {len(ordered_keys)}).')
            print(self.groups[key])



            result_path = os.path.join(self.result_folder, 'fold%03d_%s' % (idx, str(key)))
            if self.validation_dict is None:
                validation_index = (idx - 1) if idx > 0 else -1
                validation_fold_name = ordered_keys[validation_index]
                validation_set = self.groups[validation_fold_name]
                print('Test set is %s, no validation dict specified... choosing %s' % (key, validation_fold_name))
            else:
                validation_fold_name = self.validation_dict[key]
                print('Test set is %s, corresponding validation set is %s' % (key, validation_fold_name))
                validation_set = self.groups[validation_fold_name]
            train_set = self._generate_train_set(validation_fold_name, key)

            cm, mse = self._start_train(train_set, validation_set, self.groups[key], result_path, str(key), mask_postfix=mask_postfix, compute_cm=compute_cm)
            df_dict['fold'].append(key)
            df_dict['test_set'].append('_'.join(self.groups[key]))

            if compute_cm:
                prec, recall, f1, acc = compute_prec_recall_f1_acc(cm)
                df_dict['accuracy'].append(acc)
                for idy in range(len(self.mask_intervals)):
                    df_dict['precision_%d' % idy].append(prec[idy])
                    df_dict['recall_%d' % idy].append(recall[idy])
                    df_dict['f1_%d' % idy].append(f1[idy])

            if self.is_regression:
                rmse = np.sqrt(mse)
                df_dict['rmse'].append(rmse[-1])
                for idy in range(len(self.mask_intervals)):
                    df_dict['rmse_%d' % idy].append(rmse[idy])

            # End of fold
            fold_times.append(time.time() - start_time)


        df = pd.DataFrame(df_dict)
        df.to_csv(os.path.join(self.result_folder, 'report.csv'), index=False)
        with open(os.path.join(self.result_folder, 'test_validation_pairs.json'), 'w') as fp:
            json.dump(dict(self.groups), fp)
        return

    @classmethod
    def _get_intersect(cls, a, b):
        return set(a).intersection(set(b))

    def _start_train(self, train_set, validation_set: list, test_set: list, save_path: str, fold_key: str, mask_postfix='mask', compute_cm=True) -> np.ndarray:
        assert len(self._get_intersect(train_set, test_set)) == 0
        assert len(self._get_intersect(train_set, validation_set)) == 0
        assert len(self._get_intersect(validation_set, test_set)) == 0

        self._create_folder(save_path)

        print('    Training set (%d): %s' % (len(train_set), train_set))
        print('    Validation set (%d): %s' % (len(validation_set), validation_set))
        print('    Test set (%d): %s\n' % (len(test_set), test_set))
        assert len(self._get_intersect(train_set, test_set)) == 0, "Intersection of train and test set should be void"
        assert len(self._get_intersect(train_set, validation_set)) == 0, "Intersection of train and validation set should be void"
        assert len(self._get_intersect(test_set, validation_set)) == 0, "Intersection of test and validation set should be void"

        train_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.train_transforms, self.process_dict, self.csv_path, train_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)
        validation_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.test_transforms, self.process_dict, self.csv_path, validation_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)
        test_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.test_transforms, self.process_dict, self.csv_path, test_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)

        train_sampler = ShuffleSampler(train_dataset, self.seed)
        validation_sampler = ShuffleSampler(validation_dataset, self.seed)
        test_sampler = ShuffleSampler(test_dataset, self.seed)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, drop_last=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, sampler=validation_sampler, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler, drop_last=False)

        model = self._instantiate(self.model_tuple)
        initialize_weight(model, seed=self.seed)
        if torch.cuda.device_count() > 1:
            model = AttributedDataParallel(model)
        print(f"Running model to  device: {self.device}")

        model = model.to(self.device)

        criterion = self._instantiate(self.criterion_tuple)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = None
        if self.lr_scheduler_tuple is not None:
            scheduler = self.lr_scheduler_tuple[0](optimizer, **(self.lr_scheduler_tuple[1]))

        es = None
        if self.early_stop:
            es = EarlyStopping(self.patience, save_path, self.tol, verbose=True, save_best=True)

        print('    -------------------------')
        print(f'    Network training... (test on {fold_key} fold)')
        sys.stdout.flush()

        self._train(model, criterion, optimizer, train_loader, validation_loader, self.performance_eval_func, scheduler=scheduler, early_stop=es)
        print(f'    ----- Evaluation on test fold ({fold_key}) ----- ')
        cm, mse = self._validate(model, criterion, test_loader, self.performance_eval_func, compute_results=True, compute_cm=compute_cm)

        model_path = os.path.join(save_path, fold_key + '_model.pt')
        torch.save(model.state_dict(), model_path)

        pkl_path = os.path.join(save_path, fold_key + '_dict.pkl')
        mse_values = list(range(len(self.mask_intervals)))
        mse_values.append('all')
        pkl_obj = {'cm': cm, 'train_set': train_set, 'validation_set': validation_set, 'test_set': test_set, 'mse': mse, 'mse_classes': mse_values}
        with open(pkl_path, 'wb') as f:
            dump(pkl_obj, f)

        print(f'    ----- Cross-validation iteration done. ----- ')
        sys.stdout.flush()
        return cm, mse

    def _generate_train_set(self, validation_fold, test_fold):
        result = []
        assert validation_fold in self.groups
        assert test_fold in self.groups
        assert validation_fold != test_fold
        assert isinstance(validation_fold, str) and isinstance(test_fold, str)
        for grp in self.groups:
            if grp == validation_fold or grp == test_fold:
                continue
            else:
                result.extend(self.groups[grp])
        return result

    def _train(self, model, criterion, optimizer, train_loader, test_loader, performance_eval_func, n_loss_print=100, scheduler=None, early_stop=None):
        epoch_times = []
        for epoch in range(self.epochs):
            if len(epoch_times)>1:
                avg_sec = sum(epoch_times)/len(epoch_times)
                print(f"Average epoch time: {print_delta_time(avg_sec)}")
            start_time = time.time() # Measuring epoch time
            print(f"    Epoch {epoch + 1}...")

            sys.stdout.flush()
            running_loss = 0.0
            epoch_loss = 0.0
            model.train()

#             mytype = torch.float32 if isinstance(criterion, nn.MSELoss) or isinstance(criterion, nn.BCEWithLogitsLoss) else torch.long
            mytype = torch.float32 if self.is_regression else torch.long


            for idx, data in enumerate(train_loader):
                image, mask = data['image'], data['mask']
                image = image.to(self.device)
                mask = mask.to(self.device, dtype=mytype)
                if self.squeeze_mask:
                    mask = mask.squeeze(dim=1)

                optimizer.zero_grad()
                outputs = model(image)
                loss = criterion(outputs, mask)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

                if idx % n_loss_print == (n_loss_print - 1):
                    print('    Epoch: %d, Batch: %5d, Loss: %f' % (epoch + 1, idx + 1, running_loss))
                    running_loss = 0.0
            
            val_loss = self._validate(model, criterion, test_loader, performance_eval_func, early_stop=early_stop)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # End of epoch
            epoch_times.append(time.time() - start_time)

            if early_stop is not None and early_stop.early_stop:
                print('Terminating training...')
                break

        if early_stop is not None and (early_stop.early_stop or early_stop.best_loss < val_loss):
            print('Loading checkpoint because val_loss (%f) is higher than best_loss (%f)' % (val_loss, early_stop.best_loss))
            model.load_state_dict(torch.load(early_stop.save_path, map_location=self.device))
        return

    def _validate(self, model, criterion, loader, performance_eval_func, compute_results: bool=False, compute_cm=True, early_stop=None):
        """
        Print evaluation on validation set.
        Return validation loss
        """
        model.eval()
        running_loss = 0.0

        if performance_eval_func is not None and hasattr(performance_eval_func, 'reset'):
            performance_eval_func.reset()

        cm = None
        mse = None
        if compute_results:
            if compute_cm:
                cm = np.zeros((len(self.mask_intervals), len(self.mask_intervals)))
            if self.is_regression:
                sq_err = np.zeros(len(self.mask_intervals) + 1)
                counters = np.zeros(len(self.mask_intervals) + 1)

        mytype = torch.float32 if isinstance(criterion, nn.MSELoss) or isinstance(criterion, nn.BCEWithLogitsLoss) else torch.long

        with torch.no_grad():
            for idx, data in enumerate(loader):
                image, mask = data['image'], data['mask']


                image = image.to(self.device)
                mask = mask.to(self.device, dtype=mytype)
                if self.squeeze_mask:
                    mask = mask.squeeze(dim=1)

                outputs = model(image)
                
                loss = criterion(outputs, mask)
                running_loss += loss.item()

                if performance_eval_func is not None:
                    performance_eval_func(outputs, mask)

                if compute_results:
                    if self.is_regression:
                        tmp_sq_err, tmp_counters = compute_squared_errors(outputs, mask, len(self.mask_intervals))
                        sq_err += tmp_sq_err
                        counters += tmp_counters
                        if compute_cm:
                            rounded_outputs = outputs.clamp(min=0, max=(len(self.mask_intervals) - 1)).round()
                            cm += self._compute_cm(rounded_outputs, mask)
                    else:
                        if compute_cm:
                            cm += self._compute_cm(outputs, mask)

            if performance_eval_func is not None and hasattr(performance_eval_func, 'last'):
                performance_eval_func.last()

            print('    Validation loss: %f' % running_loss)

        if early_stop is not None:
            early_stop(running_loss, model)

        if compute_results:
            if self.is_regression:
                # Check for 0 values in counters, returns nan if division by zero
                mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)
            return cm, mse
        return running_loss

    def _compute_cm(self, outputs, mask):
        keepdim = mask.shape[1] == 1
        if not self.is_regression:
            if outputs.shape[1] > 1:
                prediction = outputs.argmax(axis=1, keepdim=keepdim)
            else:
                prediction = torch.sigmoid(outputs)
                prediction = torch.where(prediction > 0.5, torch.tensor(1.0, device=outputs.device), torch.tensor(0.0, device=outputs.device))
                if not keepdim:
                    prediction = prediction.squeeze(dim=1)
        else:
            prediction = outputs
            if outputs.shape[1] == 1 and (not keepdim):
                prediction = prediction.squeeze(dim=1)
        mask = mask.cpu().numpy().flatten()
        prediction = prediction.cpu().numpy().flatten()
        
        if outputs.shape[1] == 1 and not self.is_regression:
            labels = [0, 1]
        else:
            labels = list(range(len(self.mask_intervals)))
        cm = confusion_matrix(mask, prediction, labels=labels)
        return cm


class GradcamCrossValidator(CrossValidator):
    def __init__(self, groups, model_tuple, criterion_tuple, train_transforms, test_transforms, master_folder, csv_path, epochs, batch_size, lr, wd, product_list, mode, process_dict, mask_intervals, mask_one_hot, height, width, filter_validity_mask, only_burnt, mask_filtering, seed, result_folder, lr_scheduler_tuple=None, ignore_list=None, performance_eval_func=None, is_regression: bool=True, squeeze_mask=True, early_stop=False, patience=None, tol=None, threshold: float=0.5, validation_dict=None, single_fold=False):
        super().__init__(groups, model_tuple, criterion_tuple, train_transforms, test_transforms, master_folder,
                         csv_path, epochs, batch_size, lr, wd, product_list, mode, process_dict, mask_intervals,
                         mask_one_hot, height, width, filter_validity_mask, only_burnt, mask_filtering, seed,
                         result_folder, lr_scheduler_tuple=lr_scheduler_tuple, ignore_list=ignore_list,
                         performance_eval_func=performance_eval_func, squeeze_mask=squeeze_mask, early_stop=early_stop,
                         patience=patience, tol=tol, is_regression=is_regression, validation_dict=validation_dict)

        self.threshold = threshold
        self.sigm = nn.Sigmoid()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr_scheduler_tuple = lr_scheduler_tuple
        self.current_key = None
        self.model_path = None
        self.grad_results = os.path.join(result_folder, 'gradcam')
        if not os.path.isdir(self.grad_results):
            os.makedirs(self.grad_results)

    def start(self, mask_postfix='mask'):
        print('Running cross validation for Grad-CAM...')
        df_dict = {}

        ordered_keys = list(self.groups.keys())

        fold_times = [] # measuring fold computation time

        # For each fold (kfold iterations)
        for idx, key in enumerate(ordered_keys):
            print('\n_______________ ITERATION _______________')

            self.current_key = key
            if len(fold_times)>1:
                avg_sec = sum(fold_times)/len(fold_times)
                remain = predict_remaining_time(avg_sec, len(fold_times), len(ordered_keys))
                print(f"Average fold processing time: {print_delta_time(avg_sec)}, remaining time: {print_delta_time(remain)}")
            start_time = time.time() # Measuring epoch time

            print(f'Testing on {key} fold ({idx + 1} of {len(ordered_keys)}).')
            result_path = os.path.join(self.result_folder, 'fold%03d_%s' % (idx, str(key)))

            if self.validation_dict is None:
                validation_index = (idx - 1) if idx > 0 else -1
                validation_fold_name = ordered_keys[validation_index]
                validation_set = self.groups[validation_fold_name]
                print('Test set is %s, no validation dict specified... choosing %s' % (key, validation_fold_name))
            else:
                validation_fold_name = self.validation_dict[key]
                print('Test set is %s, corresponding validation set is %s' % (key, validation_fold_name))
                validation_set = self.groups[validation_fold_name]

            train_set = self._generate_train_set(validation_fold_name, key)

            # Run training for this kfold iteration, return results on test set
            df_dict[key] = self._start_train(train_set, validation_set, self.groups[key], result_path, str(key), mask_postfix=mask_postfix)


            # End of fold
            fold_times.append(time.time() - start_time)

        with open(os.path.join(self.result_folder, 'report.json'), 'w') as fp:
            json.dump(df_dict, fp)

        df = pd.DataFrame(df_dict)
        df.to_csv(os.path.join(self.result_folder, 'report.csv'), index=False)
        with open(os.path.join(self.result_folder, 'test_validation_pairs.json'), 'w') as fp:
            json.dump(dict(self.groups), fp)
        return

    def _generate_loaders(self, train_set, validation_set, test_set, mask_postfix='mask'):
        train_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height,
                                         self.width, self.product_list, self.mode, self.filter_validity_mask,
                                         self.train_transforms, self.process_dict, self.csv_path, train_set,
                                         self.ignore_list, self.mask_filtering, self.only_burnt,
                                         mask_postfix=mask_postfix)
        validation_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height,
                                              self.width, self.product_list, self.mode, self.filter_validity_mask,
                                              self.test_transforms, self.process_dict, self.csv_path, validation_set,
                                              self.ignore_list, self.mask_filtering, self.only_burnt,
                                              mask_postfix=mask_postfix)
        test_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height,
                                        self.width, self.product_list, self.mode, self.filter_validity_mask,
                                        self.test_transforms, self.process_dict, self.csv_path, test_set,
                                        self.ignore_list, self.mask_filtering, self.only_burnt,
                                        mask_postfix=mask_postfix)

        print('Train set dim: %d' % len(train_dataset))
        print('Validation set dim: %d' % len(validation_dataset))
        print('Test set dim: %d' % len(test_dataset))

        train_sampler = ShuffleSampler(train_dataset, self.seed)
        validation_sampler = ShuffleSampler(validation_dataset, self.seed)
#         test_sampler = ShuffleSampler(test_dataset, self.seed)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, drop_last=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, sampler=validation_sampler,
                                       drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        return train_loader, validation_loader, test_loader

    def _start_train(self, train_set, validation_set, test_set, save_path, fold_key, mask_postfix='mask'):
        assert len(self._get_intersect(train_set, test_set)) == 0
        assert len(self._get_intersect(train_set, validation_set)) == 0
        assert len(self._get_intersect(validation_set, test_set)) == 0

        self._create_folder(save_path)

        print('    Training set (%d): %s' % (len(train_set), train_set))
        print('    Validation set (%d): %s' % (len(validation_set), validation_set))
        print('    Test set (%d): %s\n' % (len(test_set), test_set))
        assert len(self._get_intersect(train_set, test_set)) == 0, "Intersection of train and test set should be void"
        assert len(self._get_intersect(train_set,
                                       validation_set)) == 0, "Intersection of train and validation set should be void"
        assert len(self._get_intersect(test_set,
                                       validation_set)) == 0, "Intersection of test and validation set should be void"

        # Now the validation-test-train split is completed
        train_loader, validation_loader, test_loader = self._generate_loaders(train_set, validation_set, test_set)

        model = self._instantiate(self.model_tuple)
        initialize_weight(model, seed=self.seed)

        # This is fundamental for running the CNN on CUDA
        if torch.cuda.device_count() > 1:
            model = AttributedDataParallel(model)  # Allow using multiple GPUs
        print(f"Running model to  device: {self.device}")
        model = model.to(self.device)

        criterion = self._instantiate(self.criterion_tuple)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = None
        if self.lr_scheduler_tuple is not None:
            scheduler = self.lr_scheduler_tuple[0](optimizer, **(self.lr_scheduler_tuple[1]))

        es = None
        if self.early_stop:
            es = EarlyStopping(self.patience, save_path, self.tol, verbose=True, save_best=True)

        print('    -------------------------')
        print(f'    Network training... (test on {fold_key} fold)')
        sys.stdout.flush()

        # Run training on training set, perform periodical evaluation on validation set.
        self._train(model, criterion, optimizer, train_loader, validation_loader, self.performance_eval_func, scheduler=scheduler, early_stop=es)

        # Run final validation on test set.
        print(f'    ----- Evaluation on test fold ({fold_key}) ----- ')

        self.model_path = os.path.join(save_path, fold_key + '_model.pt')
        torch.save(model.state_dict(), self.model_path)

        pkl_path = os.path.join(save_path, fold_key + '_dict.pkl')

        test_performances = self._validate_gradcam(criterion, test_loader)

        print(f'    ----- Cross-validation iteration done. ----- ')
        sys.stdout.flush()

        return test_performances

    def _validate_gradcam(self, criterion, loader):
        """
        Apply Grad-CAM algorithm on test set.
        Return IoU and CM for all the presented images
        """

        model = UNet(**self.model_tuple[1], gradcam= True)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        IOU = []
        CM = []
        for idx, data in enumerate(loader):
            bt_image, bt_mask = data['image'], data['mask']

            for i in range(bt_image.shape[0]):
                image, mask = bt_image[i, :, :, :].unsqueeze(0), bt_mask[i, :, :, :].unsqueeze(0)

                image = image.to(self.device)
                mask = mask.to(self.device, dtype=torch.float32)
                mask = np.minimum(mask, 1)

                if self.squeeze_mask:
                    mask = mask.squeeze(dim=1)

                outputs = model(image)
                bin_pred = self.sigm(outputs)
                bin_pred = torch.where(bin_pred > self.threshold, torch.tensor(1.0, device=self.device),
                                       torch.tensor(0.0, device=self.device)).squeeze().numpy()

                criterion(outputs, mask).backward()
                gradients = model.get_gradients()
                activations = model.get_activation(image)
                activations *= gradients

                heatmap = torch.mean(activations, dim=1).squeeze().detach()
                heatmap = np.maximum(heatmap, 0)
                heatmap /= heatmap.max()
                heatmap = heatmap.numpy()

                mask = mask.squeeze().numpy()
                
                img = (image.squeeze()[[3,2,1], :, :]).permute(1, 2, 0)
                img /= img.max()
                img = np.uint8(img * 255)

                cam, heatmap = self.show_cam_on_image(img, heatmap)

                IOU.append(jaccard_score(mask, bin_pred, average='micro'))
                CM.append(confusion_matrix(mask.ravel(), bin_pred.ravel()).ravel())

#                 fig, ax = plt.subplots(2, 2, figsize=(12, 12))
                # for a in (ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1]):
                #     a.set_axis_off()

                # ax[0, 0].imshow(heatmap)
                # ax[0, 0].set_title("Heatmap", y=1.0, pad=-15, color='w')

                # ax[0, 1].imshow(self.gt_on_prediction(mask, bin_pred))
                # ax[0, 1].set_title("Prediction", y=1.0, pad=-15, color='w')

                # ax[1, 0].imshow(img)
                # ax[1, 0].set_title("Sat image", y=1.0, pad=-15, color='w')

                # ax[1, 1].imshow(cam)
                # ax[1, 1].set_title("Cam result", y=1.0, pad=-15, color='w')

                # plt.subplots_adjust(wspace=0.01, hspace=0.08)

                # plt.savefig(img_path)
                # plt.close(fig)
#                 plt.axis('off')
#                 plt.imshow(self.gt_on_prediction(mask, bin_pred))
    #             plt.set_title("Cam result", y=1.0, pad=-15, color='w')
                h_path = os.path.join(self.grad_results, f"test_{self.current_key}_{idx * self.batch_size + i}-heat.png")
#                 plt.imshow(heatmap)
                pred_path = os.path.join(self.grad_results, f"test_{self.current_key}_{idx * self.batch_size + i}-pred.png")
                cv2.imwrite(pred_path, cv2.cvtColor(self.gt_on_prediction(mask, bin_pred), cv2.COLOR_RGB2BGR))
                cv2.imwrite(h_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
                


#         return {'iou' : IOU, 'cm' : CM}       
        return {'iou' : IOU}


    def _validate(self, model, criterion, loader, performance_eval_func, compute_results=False, early_stop=None):
        """
        Print evaluation on validation set.
        Return validation loss
        """
        model.eval()
        running_loss = 0.0

        # Evaluation function: (from performance_storage.py)
        if performance_eval_func is not None and hasattr(performance_eval_func, 'reset'):
            performance_eval_func.reset()

        mytype = torch.float32

        with torch.no_grad():
            for idx, data in enumerate(loader):
                image, mask = data['image'], data['mask']

                image = image.to(self.device)
                mask = mask.to(self.device, dtype=mytype)
                if self.squeeze_mask:
                    mask = mask.squeeze(dim=1)

                outputs = model(image)

                loss = criterion(outputs, mask)
                running_loss += loss.item()

                if performance_eval_func is not None:
                    performance_eval_func(outputs, mask)

                print('    Validation set results:')

                # Print evaluation results
                if performance_eval_func is not None and hasattr(performance_eval_func, 'last'):
                    performance_eval_func.last()
                # Print current validation loss
                print(f'        Loss: {running_loss}')

        if early_stop is not None:
            early_stop(running_loss, model)


        return running_loss

    @staticmethod
    def gt_on_prediction(gt, prediction):
        right = gt * prediction
        wrong = np.maximum(prediction - gt, np.zeros_like(prediction))
        missed = np.maximum(gt - prediction, np.zeros_like(prediction))

        red = right + wrong + missed
        green = right + missed
        blue = right


        image = np.array([np.where(red == 1, np.full_like(prediction, 255), np.full_like(prediction, 0)),    # RED
                 np.where(green == 1, np.full_like(prediction, 255), np.full_like(prediction, 0)),  # GREEN
                 np.where(blue == 1, np.full_like(prediction, 255), np.full_like(prediction, 0))    # BLUE
                ])
        return image.transpose(1, 2, 0)

    @staticmethod
    def show_cam_on_image(img, heatmap):
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = 1 - heatmap  # The colormap seems otherwise reversed
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        superimposed_img /= superimposed_img.max()
        superimposed_img = np.uint8(255 * superimposed_img)
        return superimposed_img, heatmap


class ConcatenatedCrossValidator(CrossValidator):
    def __init__(self, groups, model_tuple, criterion_tuple, train_transforms, test_transforms, master_folder, csv_path, epochs, batch_size, lr, wd, product_list, mode, process_dict, mask_intervals, mask_one_hot, height, width, filter_validity_mask, only_burnt, mask_filtering, seed, result_folder, lr_scheduler_tuple=None, ignore_list=None, performance_eval_func=None, is_regression: bool=True, lr2=None, squeeze_mask=True, early_stop=False, patience=None, tol=None, threshold: float=0.5, second_eval_func=None, validation_dict=None, single_fold=False):
        super().__init__(groups, model_tuple, criterion_tuple, train_transforms, test_transforms, master_folder, csv_path, epochs, batch_size, lr, wd, product_list, mode, process_dict, mask_intervals, mask_one_hot, height, width, filter_validity_mask, only_burnt, mask_filtering, seed, result_folder, lr_scheduler_tuple=lr_scheduler_tuple, ignore_list=ignore_list, performance_eval_func=performance_eval_func, squeeze_mask=squeeze_mask, early_stop=early_stop, patience=patience, tol=tol, is_regression=is_regression, validation_dict=validation_dict)
        self.lr2 = lr2 if lr2 is not None else self.lr
        self.threshold = threshold
        self.sigm = nn.Sigmoid()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.second_eval_func = second_eval_func
        self.single_fold = single_fold
        return

    def start(self, mask_postfix='mask', compute_cm=True):
        if self.single_fold:
            print('Running training...')
        else:
            print('Running cross validation...')
        df_dict = defaultdict(list)
        which = 'bin' if mask_postfix != 'mask' else 'both'

        ordered_keys = list(self.groups.keys())

        fold_times = [] # measuring fold computation time

        # For each fold (kfold iterations)
        for idx, key in enumerate(ordered_keys):
            if self.single_fold:
                if key not in {"blue"}: continue
            print('\n_______________ ITERATION _______________')
            if len(fold_times)>1:
                avg_sec = sum(fold_times)/len(fold_times)
                remain = predict_remaining_time(avg_sec, len(fold_times), len(ordered_keys))
                print(f"Average fold processing time: {print_delta_time(avg_sec)}, remaining time: {print_delta_time(remain)}")
            start_time = time.time() # Measuring epoch time

            if not self.single_fold:
                print(f'Testing on {key} fold ({idx + 1} of {len(ordered_keys)}).')
                result_path = os.path.join(self.result_folder, 'fold%03d_%s' % (idx, str(key)))
            else:
                result_path = self.result_folder

            if self.validation_dict is None:
                validation_index = (idx - 1) if idx > 0 else -1
                validation_fold_name = ordered_keys[validation_index]
                validation_set = self.groups[validation_fold_name]
                print('Test set is %s, no validation dict specified... choosing %s' % (key, validation_fold_name))
            else:
                validation_fold_name = self.validation_dict[key]
                print('Test set is %s, corresponding validation set is %s' % (key, validation_fold_name))
                validation_set = self.groups[validation_fold_name]

            train_set = self._generate_train_set(validation_fold_name, key)

            # Run training for this kfold iteration, return results on test set
            cm_bin, cm2, mse = self._start_train(train_set, validation_set, self.groups[key], result_path, str(key), mask_postfix=mask_postfix, compute_cm=compute_cm, which=which)

            # Fill in final results (dataframe for csv output)
            if not self.single_fold:
                df_dict['fold'].append(key)
                df_dict['test_set'].append('_'.join(self.groups[key]))

                if compute_cm and (which == 'both' or which == 'second'):
                    prec, recall, f1, acc = compute_prec_recall_f1_acc(cm2) # May return some nan values (division by zero)
                    df_dict['accuracy'].append(acc)
                    for idy in range(len(self.mask_intervals)):
                        df_dict['precision_%d' % idy].append(prec[idy])
                        df_dict['recall_%d' % idy].append(recall[idy])
                        df_dict['f1_%d' % idy].append(f1[idy])

                if compute_cm and (which == 'both' or which == 'bin'):
                    prec, recall, f1, acc = compute_prec_recall_f1_acc(cm_bin) # May return some nan values (division by zero)
                    df_dict['bin_accuracy'].append(acc)
                    for idy in range(2):
                        df_dict['bin_precision_%d' % idy].append(prec[idy])
                        df_dict['bin_recall_%d' % idy].append(recall[idy])
                        df_dict['bin_f1_%d' % idy].append(f1[idy])

                if self.is_regression:
                    rmse = np.sqrt(mse) # Mse is a np.array with [MSE for each class, total MSE for all classes]. May contain nan values.
                    df_dict['rmse'].append(rmse[-1])
                    for idy in range(len(self.mask_intervals)):
                        df_dict['rmse_%d' % idy].append(rmse[idy])

            # End of fold
            fold_times.append(time.time() - start_time)

        if not self.single_fold:
            df = pd.DataFrame(df_dict)
            df.to_csv(os.path.join(self.result_folder, 'report.csv'), index=False)
            with open(os.path.join(self.result_folder, 'test_validation_pairs.json'), 'w') as fp:
                json.dump(dict(self.groups), fp)
        return

    def _generate_loaders(self, train_set, validation_set, test_set, mask_postfix='mask'):
        train_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.train_transforms, self.process_dict, self.csv_path, train_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)
        validation_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.test_transforms, self.process_dict, self.csv_path, validation_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)
        test_dataset = SatelliteDataset(self.master_folder, self.mask_intervals, self.mask_one_hot, self.height, self.width, self.product_list, self.mode, self.filter_validity_mask, self.test_transforms, self.process_dict, self.csv_path, test_set, self.ignore_list, self.mask_filtering, self.only_burnt, mask_postfix=mask_postfix)

        print('Train set dim: %d' % len(train_dataset))
        print('Validation set dim: %d' % len(validation_dataset))
        print('Test set dim: %d' % len(test_dataset))

        train_sampler = ShuffleSampler(train_dataset, self.seed)
        validation_sampler = ShuffleSampler(validation_dataset, self.seed)
        test_sampler = ShuffleSampler(test_dataset, self.seed)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, drop_last=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, sampler=validation_sampler, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler, drop_last=False)

        return train_loader, validation_loader, test_loader

    def _start_train(self, train_set, validation_set, test_set, save_path, fold_key, mask_postfix='mask', compute_cm=True, which='both'):
        assert len(self._get_intersect(train_set, test_set)) == 0
        assert len(self._get_intersect(train_set, validation_set)) == 0
        assert len(self._get_intersect(validation_set, test_set)) == 0

        self._create_folder(save_path)

        print('    Training set (%d): %s' % (len(train_set), train_set))
        print('    Validation set (%d): %s' % (len(validation_set), validation_set))
        print('    Test set (%d): %s\n' % (len(test_set), test_set))
        assert len(self._get_intersect(train_set, test_set)) == 0, "Intersection of train and test set should be void"
        assert len(self._get_intersect(train_set, validation_set)) == 0, "Intersection of train and validation set should be void"
        assert len(self._get_intersect(test_set, validation_set)) == 0, "Intersection of test and validation set should be void"

        # Now the validation-test-train split is completed
        train_loader, validation_loader, test_loader = self._generate_loaders(train_set, validation_set, test_set)

        model = self._instantiate(self.model_tuple)
        initialize_weight(model, seed=self.seed)

        # This is fundamental for running the CNN on CUDA
        if torch.cuda.device_count() > 1:
            model = AttributedDataParallel(model)  # Allow using multiple GPUs
        print(f"Running model to  device: {self.device}")
        model = model.to(self.device)

        first_criterion = self._instantiate(self.criterion_tuple[0])
        optim_first = optim.Adam(model.binary_unet.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = None
        if self.lr_scheduler_tuple is not None:
            scheduler = self.lr_scheduler_tuple[0](optim_first, **(self.lr_scheduler_tuple[1]))

        es = None
        if self.early_stop:
            es = EarlyStopping(self.patience, save_path, self.tol, verbose=True, save_best=True)

        print('    -------------------------')
        print(f'    Iteration step 1/2 - Binary network training... (test on {fold_key} fold)')
        sys.stdout.flush()

        model.freeze_regression_unet()
        model.unfreeze_binary_unet()
        # Run training on training set, perform periodical evaluation on validation set.
        self._train(model, first_criterion, optim_first, train_loader, validation_loader, self.performance_eval_func, binary=True, scheduler=scheduler, early_stop=es)
        # Run final validation on test set.
        cm_bin, _, _ = self._validate(model, first_criterion, test_loader, self.performance_eval_func, binary=True, compute_results=True, compute_cm=True, which='bin')
        
        print('    -------------------------')
        print(f'    Iteration step 2/2 - Regression network training...  (test on {fold_key} fold)')
        sys.stdout.flush()

        if mask_postfix != 'mask':
            train_loader, validation_loader, test_loader = self._generate_loaders(train_set, validation_set, test_set, mask_postfix=mask_postfix)
        second_criterion = self._instantiate(self.criterion_tuple[1])

        es = None
        if self.early_stop:
            es = EarlyStopping(self.patience, save_path, self.tol, verbose=True, save_best=True)

        lr = self.lr2 if self.lr2 is not None else self.lr
        optim_second = optim.Adam(model.regression_unet.parameters(), lr=lr, weight_decay=self.wd)
        scheduler = None
        if self.lr_scheduler_tuple is not None:
            scheduler = self.lr_scheduler_tuple[0](optim_second, **(self.lr_scheduler_tuple[1]))

        model.unfreeze_regression_unet()
        model.freeze_binary_unet()
        # Run training on training set, perform periodical evaluation on validation set.
        self._train(model, second_criterion, optim_second, train_loader, validation_loader, self.second_eval_func, binary=False, scheduler=scheduler, early_stop=es)

        print(f'    ----- Evaluation on test fold ({fold_key}) ----- ')
        # Run final validation on test set.
        cm_bin2, cm2, mse = self._validate(model, second_criterion, test_loader, self.second_eval_func, binary=False, compute_results=True, compute_cm=compute_cm, which=which)

        if which == 'both':
            assert (cm_bin == cm_bin2).all()
        model_path = os.path.join(save_path, fold_key + '_model.pt')
        torch.save(model.state_dict(), model_path)

        pkl_path = os.path.join(save_path, fold_key + '_dict.pkl')
        mse_values = list(range(len(self.mask_intervals)))
        mse_values.append('all')
        pkl_obj = {'cm_bin': cm_bin, 'cm2': cm2, 'mse': mse, 'mse_classes': mse_values, 'train_set': train_set, 'validation_set': validation_set, 'test_set': test_set}
        with open(pkl_path, 'wb') as f:
            dump(pkl_obj, f)

        print(f'    ----- Cross-validation iteration done. ----- ')
        sys.stdout.flush()

        return cm_bin, cm2, mse

    def _train(self, model, criterion, optimizer, train_loader, validation_loader, performance_eval_func, binary: bool, n_loss_print=100, scheduler=None, early_stop=None):
        epoch_times = []
        # for each epoch
        for epoch in range(self.epochs):
            if len(epoch_times)>1:
                avg_sec = sum(epoch_times)/len(epoch_times)
                print(f"Average epoch time: {print_delta_time(avg_sec)}")
            start_time = time.time() # Measuring epoch time
            print(f"    Epoch {epoch + 1}...")
            sys.stdout.flush()
            running_loss = 0.0
            epoch_loss = 0.0
            model.train()

            if binary:
                model.regression_unet.eval()        # All layers in regression unet in evaluation mode
                model.freeze_regression_unet()      # All layers in regression unet are freezed (no training)
                mytype = torch.float32
            else:
                model.binary_unet.eval()
                model.freeze_binary_unet()
                mytype = torch.float32

            # for each batch in the epoch
            for idx, data in enumerate(train_loader):
                image, mask = data['image'], data['mask']

                # This is fundamental for running the CNN on CUDA
                image = image.to(self.device)
                mask = mask.to(self.device, dtype=mytype)


                if self.squeeze_mask:
                    mask = mask.squeeze(dim=1)

                optimizer.zero_grad()
                outputs = model(image)              # Run all model on the input image
                loss = criterion(outputs, mask)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

                if idx % n_loss_print == (n_loss_print - 1):
                    print('    Epoch: %d, Batch: %5d, Loss: %f' % (epoch + 1, idx + 1, running_loss))
                    running_loss = 0.0

            # End of epoch: validate on validation set (get loss and print other metrics)
            val_loss = self._validate(model, criterion, validation_loader, performance_eval_func, binary=binary, early_stop=early_stop)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # End of epoch
            epoch_times.append(time.time() - start_time)

            if early_stop is not None and early_stop.early_stop:
                print('Terminating training...')
                break

        if early_stop is not None and (early_stop.early_stop or early_stop.best_loss < val_loss):
            print('Loading checkpoint because val_loss (%f) is higher than best_loss (%f)' % (val_loss, early_stop.best_loss))
            model.load_state_dict(torch.load(early_stop.save_path, map_location=self.device))
        return

    def _validate(self, model, criterion, loader, performance_eval_func, binary: bool, compute_results=False, compute_cm=True, which='both', early_stop=None):
        """
        Print evaluation on validation set.
        Return validation loss
        """
        model.eval()
        running_loss = 0.0

        # Evaluation function: (from performance_storage.py)
        if performance_eval_func is not None and hasattr(performance_eval_func, 'reset'):
            performance_eval_func.reset()

        cm_bin = None
        cm_2 = None
        if compute_results and compute_cm:
            cm_bin = np.zeros((2, 2))
            cm_2 = np.zeros((len(self.mask_intervals), len(self.mask_intervals)))

        mse = None
        if compute_results and self.is_regression:
            sq_err = np.zeros(len(self.mask_intervals) + 1)
            counters = np.zeros(len(self.mask_intervals) + 1)

        if binary:
            mytype = torch.float32
        else:
            mytype = torch.float32

        with torch.no_grad():
            for idx, data in enumerate(loader):
                image, mask = data['image'], data['mask']

                image = image.to(self.device)
                mask = mask.to(self.device, dtype=mytype)
                if self.squeeze_mask:
                    mask = mask.squeeze(dim=1)

                outputs = model(image)
                loss = criterion(outputs, mask)
                running_loss += loss.item()

                if performance_eval_func is not None:
                    performance_eval_func(outputs, mask)

                if compute_results:
                    if compute_cm:
                        tmp_cm_bin, tmp_cm_2 = self._compute_cm(outputs, mask, which=which)
                        if which == 'both' or which == 'bin':
                            cm_bin += tmp_cm_bin
                        if which == 'both' or which == 'second':
                            cm_2 += tmp_cm_2

                    if self.is_regression:
                        # compute squared errors for each class and total pixels for each class (numpy arrays)
                        tmp_sq_err, tmp_counters = compute_squared_errors(outputs[1], mask, len(self.mask_intervals))
                        sq_err += tmp_sq_err
                        counters += tmp_counters

            print('    Validation set results:')
        
            # Print evaluation results
            if performance_eval_func is not None and hasattr(performance_eval_func, 'last'):
                performance_eval_func.last()
            # Print current validation loss
            print(f'        Loss: {running_loss}')


        if early_stop is not None:
            early_stop(running_loss, model)

        if compute_results:
            if self.is_regression:
                # Check for 0 values in counters, returns nan if division by zero
                mse = np.true_divide(sq_err, counters, np.full(sq_err.shape, np.nan), where=counters != 0)

            return cm_bin, cm_2, mse
        return running_loss

    def _compute_cm(self, outputs, mask, which='both'):
        cm_bin = None
        cm2 = None

        if which == 'bin' or which == 'both':
            if outputs[0].shape[1] == 1:
                bin_pred = self.sigm(outputs[0])
                bin_pred = torch.where(bin_pred > self.threshold, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
            else:
                bin_pred = outputs[0].argmax(dim=1, keepdim=True)
            if self.mask_one_hot:
                mask = mask.argmax(dim=1, keepdim=(not self.squeeze_mask))
            bin_mask = torch.where(mask >= 1.0, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
            cm_bin = confusion_matrix(bin_mask.cpu().numpy().flatten(), bin_pred.cpu().numpy().flatten(), labels=[0, 1])

        if which == 'second' or which == 'both':
            out2 = outputs[1]
            if self.is_regression:
                out2 = out2.round()
            else:
                out2 = out2.argmax(dim=1, keepdim=(not self.squeeze_mask))
            cm2 = confusion_matrix(mask.cpu().numpy().flatten(), out2.cpu().numpy().flatten(), labels=list(range(len(self.mask_intervals))))
        return cm_bin, cm2


class AttributedDataParallel(nn.DataParallel):
    """
    This class is required to access model attributes from DataParallel(model) without using dataparalle.module.attribute
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    # def __getattr__(self, name):
    #     attributes = vars(self.module) # List all module attributes
    #     if name in attributes:
    #         # Module attributes
    #         return getattr(self.module, name)
    #     else:
    #         # Data parallel attributes
    #         dp_attributes = vars(self)
    #         return dp_attributes[name]
