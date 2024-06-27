imp

def save_difference_map(self):
os.makedirs(self.save_pth+'diff', exist_ok=True)
if self.gt_rgb is not None:
    b, q, h, w, _3 = self.gt_rgb.shape
    for ib in range(b):
        for iq in range(q):
            filename = os.path.join(self.save_pth+'/diff/', f'rgb_{iq}.png')
            imageio.imwrite(
                filename,
                (((self.gt_rgb[ib, iq]-self.rgb[ib,iq]+1.)) *128.).detach().cpu().clamp(min=0, max=255).numpy().astype(np.uint8)
            )