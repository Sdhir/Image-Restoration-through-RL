import math
import numpy as np
import cv2
from skimage.measure import compare_mse, compare_nrmse, compare_ssim

def calculate_psnr(im, gt):
    loss = (im - gt) ** 2
    eps = 1e-10
    loss_value = loss.mean() + eps
    psnr = 10 * math.log10(1.0 / loss_value)
    return psnr

def calculate_ssim(im, gt):
    im = im[0]
    gt = gt[0]
    ssim_value = compare_ssim(im, gt, multichannel=True)
    return ssim_value

def calculate_nrmse(im, gt):
    im = im[0]
    gt = gt[0]
    nrmse_value = compare_nrmse(im, gt)
    return nrmse_value

def calculate_mse(im, gt):
    im = im[0]
    gt = gt[0]
    mse_value = compare_mse(gt, im)
    return mse_value

def load_imgs(list_in, list_gt, size = 63):
    assert len(list_in) == len(list_gt)
    img_num = len(list_in)
    imgs_in = np.zeros([img_num, size, size, 3])
    imgs_gt = np.zeros([img_num, size, size, 3])
    for k in range(img_num):
        imgs_in[k, ...] = cv2.imread(list_in[k]) / 255.
        imgs_gt[k, ...] = cv2.imread(list_gt[k]) / 255.
    return imgs_in, imgs_gt

def img2patch(my_img, size=63):
    height, width, _ = np.shape(my_img)
    assert height >= size and width >= size
    patches = []
    for k in range(0, height - size + 1, size):
        for m in range(0, width - size + 1, size):
            patches.append(my_img[k: k+size, m: m+size, :].copy())
    return np.array(patches)

def data_reformat(data):
    """RGB <--> BGR, swap H and W"""
    assert data.ndim == 4
    out = data[:, :, :, ::-1]
    out = np.swapaxes(out, 1, 2)
    return out

def sp_reward(psnr, psnr_pre):
    reward = psnr - psnr_pre
    return reward

    