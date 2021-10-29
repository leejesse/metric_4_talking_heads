import cv2
import numpy as np
from IQA_pytorch import MS_SSIM, utils
from PIL import Image
import torch
import glob
 
def averager(met, *args):
    
    metric_stack=0

    length = int(len(args[0]))
    
    gt_list = args[0]
    recon_list = args[1]
    for i in range(length):
        
        gt = cv2.imread(gt_list[i])
        recon = cv2.imread(gt_list[i])
        metric_stack += met(gt, recon) 
    
    return metric_stack/length
        

def calculate_ms_ssim(img1, img2):
    ref = utils.prepare_image(img1)
    dist = utils.prepare_image(img2)

    model = MS_SSIM(channels=3)
    
    return model(dist, ref, as_loss=False).item()

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)

    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_l1(img1=None, img2=None):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    l1 = np.mean(abs(img1-img2))

    return l1

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
gt_root = glob.glob('/home/yonggyu/pytorch-CartoonGAN/assets'+'/**/*.jpg', recursive=True)
#print(averager(calculate_ssim,gt_root, gt_root))
