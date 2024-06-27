import cv2
import numpy as np
import sys,os
import imageio
import torch
import lpips

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lpips = lpips.LPIPS(net='alex').to(device)

fn1 = sys.argv[1]
fn2 = sys.argv[2]

def get_pic_list(pic_pth):
    
    lis = os.listdir(pic_pth)
    lis = [os.path.join(pic_pth,n) for n in lis if n[:4]=='rgb_']
    return lis

if 'pts' in fn1:
    fn1 = os.path.join(fn1,'pointersect')

ls1 = get_pic_list(fn1)
ls2 = get_pic_list(fn2)

lpips_total = 0.

for p1,p2 in zip(ls1,ls2):
    img1 = cv2.imread(p1, -1)
    img2 = cv2.imread(p2, -1)
    if img1.shape[0] != img2.shape[0]:
        print(f'Resizing img1 with shape {img1.shape} to img2 with shape {img2.shape}')
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    torch_img1 = torch.from_numpy(img1).float().to(device).unsqueeze(0).permute(0,3,1,2)
    torch_img2 = torch.from_numpy(img2).float().to(device).unsqueeze(0).permute(0,3,1,2)

    lpips_score = lpips(torch_img1, torch_img2).mean()
    lpips_total += lpips_score.item()

lpips_avg = lpips_total/len(ls1)    
print(f"LPIPS between {fn1} and {fn2}: "+"{:06}".format(lpips_avg))