import cv2
import numpy as np
import sys,os
import imageio
# import package for msssim calculation
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch

fn1 = sys.argv[1]
fn2 = sys.argv[2]

def get_pic_list(pic_pth):
    
    lis = os.listdir(pic_pth)
    lis = [os.path.join(pic_pth,n) for n in lis if n[:4]=='rgb_']
    return lis
def save_difference_map(diff,save_pth,name):
    os.makedirs(save_pth+'diff', exist_ok=True)
    import IPython 
    #IPython.embed()
    filename = os.path.join(save_pth+'/diff/', name)
    dif_format = (diff+256)/2
    imageio.imwrite(
        filename,
        dif_format.astype(np.uint8),
    )
def get_color_image_msssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    # ssim_val = ssim(
    #     torch.from_numpy(img1).unsqueeze(0).permute(0,3,1,2),
    #     torch.from_numpy(img2).unsqueeze(0).permute(0,3,1,2),
    #     data_range=255, size_average=False, nonnegative_ssim=True
    # )
    msssim_val = ms_ssim(
        torch.from_numpy(img1).unsqueeze(0).permute(0,3,1,2),
        torch.from_numpy(img2).unsqueeze(0).permute(0,3,1,2),
        data_range=255
    )
    return msssim_val


if 'pts' in fn1:
    fn1 = os.path.join(fn1,'pointersect')
ls1 = get_pic_list(fn1)
ls2 = get_pic_list(fn2)
msssim_total = 0
psnr_valid_total = 0
for p1,p2 in zip(ls1,ls2):
    img1 = cv2.imread(p1, -1)
    img2 = cv2.imread(p2, -1)
    if img1.shape[0]!=img2.shape[0]:
        print(f'Resizing img1 with shape {img1.shape} to img2 with shape {img2.shape}')
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    msssim_value = get_color_image_msssim(img1, img2)
    msssim_total += msssim_value
psnr = msssim_total/len(ls1)
print(f"MS-SSIM between {fn1} and {fn2}: "+"{:06}".format(psnr))