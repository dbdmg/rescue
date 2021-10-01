import os
import re
import numpy as np
import pandas as pd

from skimage import io
from collections import defaultdict


class Scanner():
    """
    Class used to parse the folder and compute the pre-fire and post-fire dates of events. The folder must contain subfolders
    associated to satellite acquisitions and each subfolder contains images in .tiff with the following naming convention:
    product-name_yyyy_mm_dd[_cloud_coverage].tiff. The _cloud_coverage option is available only in sentinel2 products.
    The masks have the following naming convention: subfolder-name_mask.tiff.
    """
    def __init__(self, folder: str, products: list, df_path: str, mask_intervals: list, mask_one_hot: bool, ignore_list: list=None, valid_list: list=None):
        """
        Args:
            folder (str): path to master folder containing all the subfolders, related to the area of interests
            products (list(str)): list containing all the product types (i.e. 'sentinel2', 'sentinel1', 'dem', 'cloud_coverage')
            df_path (str): path to a csv containing the activation the 'folder' and 'activation_date' columns
            mask_intervals (list(tuple(int, int))): intervals used to discretize the masks
            mask_one_hot (bool): flag to return masks either with one_hot_encoding or with range [0, len(mask_intervals) - 1]
            ignore_list (list(str)): list of subfolders to ignore
            valid_list (list(str)): list of subfolders to analyze. If None, all the subfolders under 'folder' parameter are analyzed
        """
        self.folder = folder
        self.df_path = df_path
        self.products = products
        self.ignore_list = ignore_list
        self.valid_list = valid_list
        self.mask_intervals = mask_intervals
        self.mask_one_hot = mask_one_hot

        if self.df_path is not None:
            self.df = pd.read_csv(self.df_path, index_col='folder')
        else:
            self.df = None

        self.scan_master_folder()

    @classmethod
    def extract_date(cls, filename: str):
        """
        Extract the date string in the filename.
        Args:
            filename (str): filename from which the date must be extracted
        Returns:
            The date string with format 'yyyy-mm-dd' if found, None otherwise
        """
        regexp = r'.+_([0-9]{4}-[0-9]{2}-[0-9]{2})\.tiff'
        regexp = re.compile(regexp, re.IGNORECASE)

        result = regexp.search(filename)
        if result:
            return result.group(1)
        else:
            return None

    @classmethod
    def convert_date(cls, date: str):
        """
        Converts the date string in format 'dd/mm/yyyy' in 'yyyy-mm-dd'.
        Args:
            date (str): date string to be converted
        Returns:
            The date string with format 'yyyy-mm-dd'
        """
        split = date.split('/')
        split.reverse()
        new_date = '-'.join(split)
        return new_date

    def scan_folder(self, folder: str) -> dict:
        """
        Scan the content of specified subfolder to identify all the products of interest files (element of self.products).
        For each product, the files are analyzed to find the pre-fire and post-fire acquisition dates.

        Args:
            folder (str): fullpath of the folder to be analyzed
        Returns:
            dict(dict(str)): returns a dict containing for each product a dict with two keys: 'pre' and 'post'.
            If the value associated to 'pre' or 'post' is empty, the file was not found. To decide whether an acquisition
            is pre-fire or post-fire, the min and max dates are computed. If only one single file was found for a product,
            if df_path was specified, the activation_date value is used. Otherwise, the file is associated to post-fire.
        """
        result = {}
        files = defaultdict(list)
        _, foldername = os.path.split(folder)

        if not os.path.isdir(folder):
            raise ValueError('Invalid folderpath specified %s' % folder)

        for filename in os.listdir(folder):
            filename_fullpath = os.path.join(folder, filename)
            if not os.path.isfile(filename_fullpath):
                continue

            date = self.extract_date(filename)
            if date is not None:
                for prod in self.products:
                    if prod == 'cloud_coverage':
                        if 'sentinel2' in self.products:
                            continue
                        else:
                            prod = 'sentinel2'
                    if filename.startswith(prod):
                        files[prod].append(date)
                        break

        for prod in self.products:
            if prod == 'cloud_coverage':
                if 'sentinel2' in self.products:
                    continue
                else:
                    prod = 'sentinel2'
            result[prod] = {}
            if len(files[prod]) > 1:
                result[prod]['pre'] = min(files[prod])
                result[prod]['post'] = max(files[prod])
            elif len(files[prod]) == 1:
                if self.df is None:
                    result[prod]['pre'] = ''
                    result[prod]['post'] = files[prod][0]
                else:
                    activation_date = self.convert_date(self.df.loc[foldername]['activation_date'])
                    if files[prod][0] > activation_date:
                        result[prod]['pre'] = ''
                        result[prod]['post'] = files[prod][0]
                    else:
                        result[prod]['post'] = ''
                        result[prod]['pre'] = files[prod][0]
            elif len(files[prod]) == 0:
                print('Empty folder found %s for product %s' % (folder, prod))
                result[prod]['pre'] = ''
                result[prod]['post'] = ''

        return result

    def scan_master_folder(self): 
        """
        Scan the master folder to compute self.dates_dict. This dictionary contains for each subfolder the result of scan_folder method.
        """    
        self.dates_dict = {}  
        if not os.path.isdir(self.folder):
            raise ValueError('Invalid folderpath value %s' % self.folder)

        for dirname in os.listdir(self.folder):
            dirpath = os.path.join(self.folder, dirname)
            
            if self.ignore_list is not None and dirname in self.ignore_list:
                continue

            if self.valid_list is not None and dirname not in self.valid_list:
                continue

            if not os.path.isdir(dirpath):
                continue

            self.dates_dict[dirname] = self.scan_folder(dirpath)

        return

    def get(self, folder: object, product: object, mode: object) -> object:
        """
        Get the requested product in the specified mode (pre-fire or post-fire) for the specified subfolder.
        Args:
            folder (str): subfolder name of the information to be retrieved
            product (str): product of interest (i.e. 'sentinel2', 'sentinel1', 'dem', 'cloud_coverage')
            mode (str): either 'pre' or 'post'

        Returns:
            np.ndarray: the requested image
        """
        if mode != 'pre' and mode != 'post':
            raise ValueError('Invalid mode parameter specified %s' % mode)

        if product != 'cloud_coverage':
            tmp_product = product
        else:
            tmp_product = 'sentinel2'
        date = self.dates_dict[folder][tmp_product][mode]

        if product != 'cloud_coverage':
            filename = tmp_product + '_' + date + '.tiff'
        else:
            filename = tmp_product + '_' + date + '_cloud_coverage.tiff'
        path = os.path.join(self.folder, folder, filename)

        if not os.path.isfile(path):
            print('Invalid product requested:')
            print('root: %s' % self.folder)
            print('folder: %s' % folder)
            print('product: %s' % product)
            print('mode: %s' % mode)
            print('date: %s' % date)
            raise RuntimeError('File not found %s' % path)

        result = io.imread(path)
        if len(result.shape) == 2:
            result = result[..., np.newaxis]
        return result

    def get_mask(self, folder: str, mask_postfix: str='mask', apply_mask_discretization: bool=True) -> np.ndarray:
        """
        Get the severity mask associated to the subfolder of interest.

        Args;
            folder (str): subfolder name of interest
            mask_postfix (str): used to retrieve mask tiff. The file 'folder_(mask_postfix).tiff' is loaded
            apply_mask_discretization (bool): if True apply mask intervals to mask image to discretize it.
        Returns:
            np.ndarray: the requested mask, either with one_hot_encoding or not depending on the parameter specified in the constructor.
        """
        filename = folder + '_%s.tiff' % mask_postfix
        path = os.path.join(self.folder, folder, filename)

        if not os.path.isfile(path):
            print('Invalid mask requested:')
            print('root: %s' % self.folder)
            print('folder: %s' % folder)
            print('filename: %s' % filename)
            raise RuntimeError('Mask file not found %s' % path)

        mask = io.imread(path)
        if apply_mask_discretization:
            result = np.zeros_like(mask) if not self.mask_one_hot else np.zeros((mask.shape[0], mask.shape[1], len(self.mask_intervals)))
            for idx, (lower, higher) in enumerate(self.mask_intervals):
                bin_mask = (mask >= lower) & (mask <= higher)
                if self.mask_one_hot:
                    result[bin_mask, idx] = 1
                else:
                    result[bin_mask] = idx
                    
            if not self.mask_one_hot:        
                result = result[..., np.newaxis]
        else:
            result = mask
            if len(result.shape) == 2:
                result = result[..., np.newaxis]
        return result

    @classmethod
    def _generate_mode_dict(cls, product_list: list, mode: str, pre_before :bool=True) -> dict:
        """
        Internal function to be used in case mode parameter is simply a string to be applied to all the product list.

        Args:
            product_list (list(str)): list of products
            mode (str): either 'pre', 'post' or 'both'
            pre_before (bool): default is True. If True, returns the pre-fire image before the post-fire image

        Returns:
            dict: for each product (key), a list containing the specified modes
        """
        result = {}

        for product in product_list:
            if mode != 'both':
                result[product] = [mode]
            else:
                if pre_before:
                    result[product] = ['pre', 'post']
                else:
                    result[product] = ['post', 'pre']

        return result

    def get_all(self, folder: str, product_list: list, mode, retrieve_mask: bool=True, mask_postfix: str='mask', apply_mask_discretization: bool=True) -> tuple:
        """
        Gets all the products with all the specified modes from subfolder.

        Args:
            folder (str): subfolder name from which the products must be collected
            product_list (list(str)): list of products to retrieve
            mode (str or dict): either a string ('both', 'pre', 'post') or a dict containing for each product a list of modes
            retrieve_mask (bool): default is True. If True, collect as last image the severity mask from folder.
            mask_postfix (str): used to retrieve mask as 'folder_(mask_postfix).tiff'.
            apply_mask_discretization (bool): if True apply mask intervals to mask image to discretize it.
        Returns:
            tuple(list(np.ndarray), list(str)): lists of images with the corresponding product names with modes (e.g. 'sentinel2_pre', 'sentinel1_post', 'mask')
        """
        if mode != 'both' and mode != 'pre' and mode != 'post' and not isinstance(mode, dict):
            raise ValueError('Invalid value specified for mode: %s' % mode)

        if isinstance(mode, dict):
            mode_dict = mode
        else:
            mode_dict = self._generate_mode_dict(product_list, mode)

        img_result = []
        product_order = []
        for product in product_list:
            for current_mode in mode_dict[product]:
                img = self.get(folder, product, current_mode)
                img_result.append(img)
                product_order.append(product + '_' + current_mode)

        if retrieve_mask:
            img = self.get_mask(folder, mask_postfix=mask_postfix, apply_mask_discretization=apply_mask_discretization)
            img_result.append(img)
            product_order.append('mask')
        
        return img_result, product_order