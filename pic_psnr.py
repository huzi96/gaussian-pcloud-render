import cv2
import numpy as np
import sys,os
import imageio

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
if 'pts' in fn1:
    fn1 = os.path.join(fn1,'pointersect')

ls1 = get_pic_list(fn1)
ls2 = get_pic_list(fn2)
psnr_total = 0
psnr_valid_total = 0
for p1,p2 in zip(ls1,ls2):
    img1 = cv2.imread(p1, -1)
    img2 = cv2.imread(p2, -1)
    if img1.shape[0]!=img2.shape[0]:
        print(f'Resizing img1 with shape {img1.shape} to img2 with shape {img2.shape}')
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse = np.mean(diff ** 2)
    non_zero_img2_mask = img2 != 0
    valid_diff = diff * non_zero_img2_mask
    valid_mse = np.sum(valid_diff ** 2) / np.sum(non_zero_img2_mask)

    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    psnr_total += psnr
    save_difference_map(diff,'validate/res/render/difmap2/',p1[-6:])

    psnr_valid = 20 * np.log10(255) - 10 * np.log10(valid_mse)
    psnr_valid_total += psnr_valid

psnr = psnr_total/len(ls1)
print(f"psnr between {fn1} and {fn2}: "+"{:06}".format(psnr))
# psnr_valid = psnr_valid_total/len(ls1)
# print(f"valid psnr between {fn1} and {fn2}: "+"{:06}".format(psnr_valid))