import re
import math
import numpy as np

from collections import defaultdict
from skimage.transform import resize, rescale

class ProductProcessor():
    """
    Class used to process products within the same folder and eventually combine pre-fire and post-fire data according to a proper function.
    """
    def __init__(self, image_transfomer, combiner, name: str, pre_first: bool=False):
        """
        Constructor of ProductProcessor.

        Args:
            image_transformer (callable): a function which accepts an image as input (ndarray) and returns an image (ndarray). Applied to both pre-fire and         post-fire image. May be None: in this case, images are passed directly to combiner, if present
            combiner (callable): a function which accept pre-fire and post-fire image as inputs and returns an image as output. It is used to combine pre-fire and post-fire image into a single product. May be None. In that case, pre-fire and post-fire image are returned concatenated along the channel axis, if both are present, otherwise a single one is returned.
            name (str): name of the returned final product, used internally to store information in the product list
            pre_first (bool): if True, concatenates pre-fire image before the post-fire image along the channel axis. Used only if combiner is None
        """
        self.image_transformer = image_transfomer
        self.combiner = combiner
        self.pre_first = pre_first

        if name is not None and name != '':
            self.name = name
        else:
            first = '' if self.image_transformer is None else self.image_transformer.__name__
            second = '' if self.combiner is None else self.combiner.__name__
            self.name = first + '_' + second
        return

    def __call__(self, pre: np.ndarray, post: np.ndarray) -> tuple:
        """
        Used to perform all transform operation on pre-fire and post-fire image. First, if self.image_transformer is not None, pre-fire and post-fire image are transformed. Then, if combiner is not None, the (transformed) pre-fire and post-fire images are combined together by calling the combiner.

        Args:
            pre (np.ndarray): pre-fire image
            post (np.ndarray): post-fire image

        Return:
            tuple(ndarray, str): returns a tuple (image, name), where image is the image transformed by eventually combining together image_transformer and combiner, if present. If combiner is absent, pre-fire and post-fire are concatenated accordingly if both of them are present. The name (self.name) is used to describe the returned image
        """
        if self.image_transformer is not None:
            if callable(self.image_transformer):
                pre = self.image_transformer(pre) if pre is not None else None
                post = self.image_transformer(post) if post is not None else None
            elif isinstance(self.image_transformer, list):
                pre = pre[:, :, self.image_transformer] if pre is not None else None
                post = post[:, :, self.image_transformer] if post is not None else None
            else:
                raise ValueError('Invalid image_transformer specified')
        if self.combiner is not None:
            result = self.combiner(pre, post)
        else:
            if pre is not None and post is not None:
                result = np.concatenate([pre, post], axis=-1) if self.pre_first else np.concatenate([post, pre], axis=-1)
            elif pre is None and post is not None:
                result = post
            elif pre is not None and post is None:
                result = pre
            else:
                raise ValueError('Pre and post images are both None')

        result = self.expand_axis(result)
        return result, self.name

    @classmethod
    def expand_axis(cls, img):
        """
        Class method used to expand 2D, single-channel images to ndarray with 3 dimensions: H x W x 1.

        Args:
            img (np.ndarray): input image
        Returns:
            np.ndarray: the expanded image
        """
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                img = img[..., np.newaxis]
        elif isinstance(img, list):
            for idx, element in enumerate(img):
                img[idx] = cls.expand_axis(element)

        return img


class ImageProcessor():
    """
    Class used to process images from the same area of interest. Through this class, it is possible to upscale the image
    to the highest resolution among all the products, cut the images into tiles and process the single products through
    instances of the ProductProcessor class.
    """
    def __init__(self, height: int, width: int) -> object:
        """
        Constructor of ImageProcess class.

        Args:
            height (int): height of the single tile of the images
            width (int): width of the single tile of the images
        """
        self.height = height
        self.width = width
        self.dimension_dict = defaultdict(dict)
        return

    @classmethod
    def _generate_product_tree(cls, product_list):
        """
        Auxiliary function used to generate nested dictionary to easily retrieve indices from the product_list. The
        dictionary contains for each product and each mode ('pre' or 'post', except for 'mask') the index to collect the
        product with the desired mode from product_list.

        Args:
            product_list (list(str)): list of products (e.g. ['sentinel2_post', 'sentinel2_pre', 'sentinel1_pre', 'mask'])

        Returns:
            dict: dictionary containing for each key (e.g. 'sentinel2') a subdictionary ('pre' and/or 'post') with the
            corresponding index. 'mask' is the only product without mode
        """
        result = defaultdict(dict)
        regexp = r'^(((?!pre|post).)+)(_(pre|post))?$'
        regexp = re.compile(regexp, re.IGNORECASE)

        for idx, product in enumerate(product_list):
            regexp_result = regexp.search(product)
            if not regexp_result:
                raise ValueError('Invalid product specified %s' % product)
            product_name = regexp_result.group(1)
            mode = regexp_result.group(4)
            assert mode == 'pre' or mode == 'post' or mode is None
            if mode is not None:
                result[product_name][mode] = idx
            else:
                result[product_name] = idx

        return dict(result)

    @classmethod
    def _extract_image_from_array(cls, images, offset, n_channels):
        """
        Extracts the desired image from the numpy array, which contains the images of all the products concatenated
        along the channel axis.
        e.g. products = ['sentinel2_post' (13 channels), 'sentinel1_pre' (4 channels), 'dem_pre' (2 channels)]. -> images.shape = [n_images x] height x width x 19 (13 + 4 + 2)

        Args:
            images (np.ndarray): images of all the products concatenated along the channel axis
            offset (int): channel axis from which the image must be extracted
            n_channels (int): number of continguous channels to be extracted

        Returns:
            np.ndarray: the extracted image.
        """
        if offset < 0 or n_channels < 0:
            return None

        result = images[..., :, :, offset:(offset + n_channels)]
        return result

    @classmethod
    def _compute_offset_channels(cls, product_tree, channel_counter, product, mode):
        """
        Given a product and its mode (i.e. 'pre' or 'post'), computes the offset and the number of channels

        Args:
            product_tree (dict): 2-level nested dictionary which contains for each key, the index values of the product
            and mode of interest from which the information must be retrieved channel_counter (list(int)): list containing at each cell, for each product, the number of channels of corresponding product.
                                        e.g. product_list = ['sentinel2', 'sentinel1', 'mask'] -> channel_counter = [13, 4, 1]
            product (str): product of interest
            mode (str): mode of interest. This parameter is ignored in case product == 'mask'

        Returns:
            tuple(int, int): offset and n_channels used to extract the image from the images array (where all the products are concatenated along the channel axis)
        """
        index = -1
        if product == 'mask':
            index = product_tree[product]
        elif mode in product_tree[product]:
            index = product_tree[product][mode]

        if index < 0:
            return -1, -1

        offset = sum(channel_counter[:index])
        n_channels = channel_counter[index]
        return offset, n_channels

    def process(self, img_list, product_list: list, process_dict: dict, output_order: list=None, pre_first: bool=False, channel_counter: list=None, return_ndarray=False) -> tuple:
        """
        Process method used to apply the ProductProcessor at each product, to both the pre-fire and post-fire image (if present).

        Args:
            img_list (Union[list, np.ndarray]): either a python list of np.ndarray (images) or nd.array, where all the products are concatenated on the channel axis
            product_list (list(str)): list of str which specify for each image in img_list (or images concatenated along the channel axis), the product type (e.g. 'sentinel2')
            process_dict (dict): dictionary defining for each product (e.g. 'sentinel1', 'dem') a list of ints (channels to be selected) or a ProductProcessor to process the data for that specific product (specified as key).
                                e.g. {'sentinel2': [3, 2, 1], 'sentinel1': ProductProcessor(...)} -> sentinel2 products (both pre and post-fire if present) are filtered by collecting only channels [3, 2, 1], whereas 'sentinel1' products are modified by executing the defined ProductProcessor.
            output_order (list(str)): order of output products (e.g. ['sentinel2', 'sentinel1', 'dem']). Default is None.
            pre_first (bool): In case a combiner is not specified and both pre-fire and post-fire images are present, returns the two images concatenated along the channel axis, with the pre-fire image first. Default is False.
            channel_counter (list(int)): list of ints, which specifies for each product in product_list the number of channels for that specified product. Parameter is ignored if img_list is a python list and used only if img_list is a np.ndarray.
                                e.g. product_list = ['sentinel2', 'sentinel1'], channel_counter = [13, 4] -> sentinel2 products have 13 channels, sentinel1 products have 4 channels

        Returns:
            tuple(Union[list, np.ndarray], str): returns a python list (if img_list is a python list) or a np.ndarray (if img_list is np.ndarray) containing all the processed images. The returned string is a name describing the returned product
        """
        if not (isinstance(img_list, list) or (isinstance(img_list, np.ndarray) and channel_counter is not None)):
            raise ValueError('Invalid set of parameters. Either img_list must be list or img_list must be np.array with channel_counter specified')

        if channel_counter is not None:
            assert len(channel_counter) == len(product_list)
        
        product_tree = self._generate_product_tree(product_list)
        result = []
        result_name = []

        regexp = r'^(((?!pre|post).)+)(_(pre|post))?$'
        regexp = re.compile(regexp, re.IGNORECASE)

        if output_order is None:
            output_order = []
            for prod in product_list:
                if prod == 'mask':
                    continue
                regexp_result = regexp.search(prod)
                if regexp_result:
                    val = regexp_result.group(1)
                    if val not in output_order:
                        output_order.append(val)
                else:
                    raise ValueError('Product %s did not match with regexp' % prod)

        out_order_len = len(output_order) if 'mask' not in output_order else len(output_order) - 1
        if len(process_dict) != out_order_len:
            raise ValueError('Invalid output_order specified: %s - Process dict keys not matching: %s' % (str(output_order), str(list(process_dict.keys()))))
        for out in output_order:
            if out not in process_dict:
                raise ValueError('Invalid product name in output_order not found in process_dict: %s' % out)

        for idx, out in enumerate(output_order):
            if out == 'mask':
                continue
            processor = process_dict[out]
            pre_index = product_tree[out]['pre'] if 'pre' in product_tree[out] else -1
            post_index = product_tree[out]['post'] if 'post' in product_tree[out] else -1

            if isinstance(img_list, list):
                pre = img_list[pre_index] if pre_index != -1 else None
                post = img_list[post_index] if post_index != -1 else None
            elif isinstance(img_list, np.ndarray):
                pre_offset, pre_channels = self._compute_offset_channels(product_tree, channel_counter, out, 'pre')
                post_offset, post_channels = self._compute_offset_channels(product_tree, channel_counter, out, 'post')

                pre = self._extract_image_from_array(img_list, pre_offset, pre_channels)
                post = self._extract_image_from_array(img_list, post_offset, post_channels)
            if isinstance(processor, list):
                pre = pre[:, :, processor] if pre_index != -1 else None
                post = post[:, :, processor] if post_index != -1 else None
                if pre is None and post is not None:
                    tmp = post
                elif pre is not None and post is None:
                    tmp = pre
                else:
                    tmp = np.concatenate([pre, post], axis=-1) if pre_first else np.concatenate([post, pre], axis=-1)
                tmp_name = out
            elif isinstance(processor, ProductProcessor):
                tmp, tmp_name = processor(pre, post)
            else:
                raise ValueError('Invalid processor parameter: %s' % str(processor))

            if isinstance(tmp, list):
                result.extend(tmp)
                result_name.extend(tmp_name)
            else:
                result.append(tmp)
                result_name.append(tmp_name)

        if 'mask' not in output_order and 'mask' in product_list:
            if isinstance(img_list, list):
                mask = img_list[product_tree['mask']]
            else:
                mask_offset, mask_channels = self._compute_offset_channels(product_tree, channel_counter, 'mask', '')
                mask = self._extract_image_from_array(img_list, mask_offset, mask_channels)
            result.append(mask)
            result_name.append('mask')

        if return_ndarray:
            result = np.concatenate(result, axis=-1)

        return result, result_name

    def upscale(self, img_list: list, product_list: list, foldername: str, concatenate: bool):
        """
        Upscales all the images to the highest resolution among all the images. Images are assumed to always have the same aspect ratio.

        Args:
            img_list (list(np.ndarray)): list of np.ndarray containing all the images
            product_list (list(str)): list of string, each providing a description/product name for each image in img_list
            foldername (str): foldername from which all the products where loaded. Used to populate the dimension_dict, an internal dictionary used to store all the height and width for each upscaled product
            concatenate (bool): if True, returns a np.ndarray containing all the upscaled images concatenated along the channel axis. Otherwise, returns a python list.

        Returns:
            Union(list, np.ndarray): if concatenate is True, a np.ndarray image is returned containing all the images concatenated along the channel axis, otherwise a python list containing all the images is returned.
        """
        result = []
        assert len(img_list) == len(product_list)

        max_height = -1
        max_width = -1
        max_product = ''
        for product, img in zip(product_list, img_list):
            if img.shape[0] > max_height and img.shape[1] > max_width:
                max_height = img.shape[0]
                max_width = img.shape[1]
                max_product = product

        self.dimension_dict[foldername]['height'] = max_height
        self.dimension_dict[foldername]['width'] = max_width
        self.dimension_dict[foldername]['product'] = max_product

        for img in img_list:
            if img.shape[0] != max_height and img[1] != max_width:
                img = resize(img, (max_height, max_width))
            result.append(img)
        if not concatenate:
            return result
        else:
            return np.concatenate(result, axis=-1)

    @classmethod
    def _reshape_min(cls, img: np.ndarray, height: int, width: int, contain_mask: bool, apply_mask_round: bool):
        """
        Rescale the image such that resulting_image.height >= height and resulting_image.width >= width

        Args:
            img (np.ndarray): image to be rescaled
            height (int): minimum height value
            width (int): minimum width value
            contain_mask (bool): True if img contains a mask. If True and img need to be upscaled, the mask (last channel axis) is rounded to the nearest integer value.
            apply_mask_round (bool): if True and contain_mask is True, the mask is rounded after the cut.
        Returns:
            np.ndarray: the upscaled image
        """
        res = img.shape[-3:]
        rescale_height = height / res[0]
        rescale_width = width / res[1]

        if rescale_height <= 1 and rescale_width <= 1:
            return img

        result = rescale(img, max(rescale_height, rescale_width), multichannel=True)
        if contain_mask and apply_mask_round:
            result[..., -1] = np.rint(result[..., -1])
        return result

    def cut(self, image, product_list: list, return_ndarray=False, apply_mask_round: bool=True):
        """
        Method used to cut the images into smaller tiles of shape self.height x self.width. In case the image is not a multiple of self.height and self.width, the last images are overlapped with the previous ones.

        Args:
            image (Union[list, np.ndarray]): python list containing all the images to be cut or np.ndarray with all the products concatenated along the channel axis
            product_list (list): list of products. Used to detect whether the mask is in the list or not and round it in case its resolution is lower than self.height and self.width and an upscale operation is needed.
        Returns:
            Union[list, np.ndarray]: returns either a list (if image is a list) or np.ndarray (if image is np.ndarray) containing all the tiles obtained by cutting the image. Resulting shape is:
                list -> n_images: each image is n_tiles x self.height x self.width x channels
                np.ndarray -> n_cut x self.height x self.width x total_channels
        """
        def _cut(image, height, width, contain_mask, apply_mask_round):
            image = self._reshape_min(image, height, width, contain_mask, apply_mask_round)
            resolution = image.shape 
            resolution = resolution[-3:] #height x width x channels
            max_i = math.ceil(resolution[0] / self.height)
            max_j = math.ceil(resolution[1] / self.width)

            result = []
            for i in range(max_i):
                for j in range(max_j):
                    vertical_slice = slice(min(height * i, resolution[0] - height), min(height * (i + 1), resolution[0]), 1)
                    horizontal_slice = slice(min(width * j, resolution[1] - width), min(width * (j + 1), resolution[1]), 1)
                    tmp = image[..., vertical_slice, horizontal_slice, :]

                    assert tmp.shape[-3:][0] == height and tmp.shape[-3:][1] == width and tmp.shape[-3:][2] == resolution[2]

                    result.append(tmp)

            result = np.array(result) #n_cuts x [n_images x] height x width x channels
            result = np.swapaxes(result, 0, -4) #[n_images x] n_cuts x height x width x channels
            return np.array(result), max_i * max_j

        if isinstance(image, list):
            result = []
            count_result = []
            for img, prod in zip(image, product_list):
                contain_mask = prod == 'mask'
                tmp, cnt = _cut(img, self.height, self.width, contain_mask, apply_mask_round)
                result.append(tmp)
                count_result.append(cnt)
            if return_ndarray:
                result = np.concatenate(result, axis=-1)
            return result, count_result
        elif isinstance(image, np.ndarray):
            contain_mask = 'mask' in product_list
            return _cut(image, self.height, self.width, contain_mask, apply_mask_round)
        else:
            raise ValueError('Invalid data type of image %s' % str(type(image)))