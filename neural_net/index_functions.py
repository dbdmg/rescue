import os
import numpy as np

def filter_invalid_values(img, default_value='min', corrupted_mask=None):
    mask = (~np.isnan(img)) & (~np.isinf(img))
    if corrupted_mask is not None:
        crp = corrupted_mask
        if len(corrupted_mask.shape) == 3:
            crp = corrupted_mask.squeeze(axis=-1)
        if len(mask.shape) == 3:
            mask = mask.squeeze(axis=-1)
        mask = mask & crp
    if isinstance(default_value, float) or isinstance(default_value, int):
        img[np.isnan(img)] = default_value
    elif default_value == 'mean':
        img[np.isnan(img)] = img[mask].mean()
    elif default_value == 'median':
        img[np.isnan(img)] = img[mask].median()
    elif default_value == 'max':
        img[np.isnan(img)] = img[mask].max()
    elif default_value == 'min':
        img[np.isnan(img)] = img[mask].min()
    else:
        raise ValueError('Invalid value for default_value %s' % str(default_value))
    img[np.isposinf(img)] = img[mask].max()
    img[np.isneginf(img)] = img[mask].min()
    return img

def filter_s2_image(img, mask, default_value='min'):
    mask2 = mask == 0
    invalid_mask = (~np.isnan(img)) & (~np.isinf(img))
    if isinstance(default_value, float) or isinstance(default_value, int):
        img[mask2] = default_value
    elif default_value == 'mean':
        img[mask2] = img[(~mask2) & invalid_mask].mean()
    elif default_value == 'median':
        img[mask2] = img[(~mask2) & invalid_mask].median()
    elif default_value == 'max':
        img[mask2] = img[(~mask2) & invalid_mask].max()
    elif default_value == 'min':
        img[mask2] = img[(~mask2) & invalid_mask].min()
    else:
        raise ValueError('Invalid parameter for default_value %s' % str(default_value))

    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    return img

def _diff_sum(img, b1, b2, filter_invalid=True, eps=1e-5):
    num = img[:, :, b1] - img[:, :, b2]
    den = img[:, :, b1] + img[:, :, b2]
    result = num / (den + eps)
    if len(img.shape) == 13:
        result = filter_s2_image(result, img[:, :, 12])
    if filter_invalid:
        corrupted_mask = img[:, :, -1] == 1
        filter_invalid_values(result, corrupted_mask=corrupted_mask)
    return result

def nbr(img, filter_invalid=False, zero_nan=True, eps=1e-5):
    #(B08 - B12)/(B08 + B12)
    result = _diff_sum(img, 7, 11, filter_invalid=filter_invalid, eps=eps)
    if zero_nan:
        mask = np.isnan(result)
        result[mask] = 0
    return result   

def nbr2(img, filter_invalid=False):
    #(B11 - B12)/(B11 + 12)
    return _diff_sum(img, 10, 11, filter_invalid=filter_invalid)

def ndvi(img, filter_invalid=False):
    #(B08 - B04) / (B08 + B04)
    return _diff_sum(img, 7, 3, filter_invalid=filter_invalid)

def gndvi(img, filter_invalid=False):
    #(B08 - B03) / (B08 + B03)
    return _diff_sum(img, 7, 2, filter_invalid=filter_invalid)

def clre(img, filter_invalid=False):
    #(B7 / B5) - 1
    result = (img[:, :, 6] / img[:, :, 4]) - 1
    result = filter_s2_image(result, img[:, :, 12])
    return result

def ndre2(img, filter_invalid=False):
    # (B7 - B5) / (B7 + B5)
    result = _diff_sum(img, 6, 4, filter_invalid=filter_invalid)
    return result

def ndre1(img, filter_invalid=False):
    # (B6 - B5) / (B6 + B5)
    result = _diff_sum(img, 5, 4, filter_invalid=filter_invalid)
    return result

def bais2(img, filter_invalid=False):
    #(1 - sqrt(B06 * B07 * B8A / B04)) * ((B12 - B8A) / sqrt(B12 + B8A) + 1)
    sqrt = np.sqrt((img[:, :, 5] * img[:, :, 6] * img[:, :, 8]) / img[:, :, 3])
    term2 = ((img[:, :, 11] - img[:, :, 8]) / np.sqrt(img[:, :, 11] + img[:, :, 8])) + 1
    result = (1 - sqrt) * term2
    result = filter_s2_image(result, img[:, :, 12])
    if filter_invalid:
        filter_invalid_values(result)
    return result

def bai(img, filter_invalid=False):
    # 1 / ((0.1 - B04) ** 2 + (0.06 - B08) ** 2)
    t1 = 0.1 - img[:, :, 3]
    t1 = t1 * t1
    t2 = 0.06 - img[:, :, 7]
    t2 = t2 * t2
    result = (1 / (t1 + t2))
    result = filter_s2_image(result, img[:, :, 12])
    return result

def custom_post(img, filter_invalid=True):
    result = []
    result.append(img[..., [0, 1, 2, 3, 7, 8, 10, 11]])
    result.append(nbr(img, filter_invalid))
    result.append(nbr2(img, filter_invalid))
    result.append(ndvi(img, filter_invalid))
    result.append(bais2(img, filter_invalid))

    result = np.concatenate(result, axis=-1)
    return result

def burned_index_only(img, filter_invalid=True):
    result = []
    result.append(nbr2(img, filter_invalid))
    result.append(ndvi(img, filter_invalid))

    result = np.concatenate(result, axis=-1)
    return result