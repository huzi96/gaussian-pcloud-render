import subprocess
def rescale_run(input,output,factor, input_offset, show=False, offset=512):
    commend = f"python validate/code/util_rescale_ply.py \
                --input_ply {input} \
                --output_ply {output} \
                --factor {factor} \
                --input_offset {input_offset} \
                --offset {offset}"
    print('Executing: ', commend)

    subp=subprocess.Popen(commend,shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return 

def scale_run(input,output,factor, show=False):
    commend = f"python validate/code/util_scale_ply.py \
                --input_ply {input} \
                --output_ply {output} \
                --factor {factor}"
    subp=subprocess.Popen(commend,shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return  

def pts_run(input, output,setting=(50,800,600), mode='udlrfb',img_n=6, show=False, background_color=0., cam_info=None):
    if cam_info is None:
        setting_str = f"'fov':{setting[0]},'width_px':{setting[1]},'height_px':{setting[2]}"
        setting_str = '{'+setting_str+'}'
    else:
        setting_str = str(cam_info).replace('\"', '\'')
        setting_str = setting_str.replace(' ', '')
    # commend = "python pointersect/inference/main.py \
    #         --input_point_cloud "+input+"\
    #         --output_dir "+output+" --render_poisson 0 \
    #         --save_settings '{\"background_color\":0}'\
    #         --output_camera_trajectory_mode "+mode+"\
    #         --output_camera_setting '{"+setting_str+"}' \
    #         --k 40 --n_output_imgs "+str(img_n)
    # print(commend)
    command = ['python','pointersect/inference/main.py',
               '--input_point_cloud', input,
               '--output_dir', output,
               '--render_poisson', '0',
               '--save_settings', '{\"background_color\":'+str(background_color)+'}',
               '--output_camera_trajectory_mode', mode,
               '--output_camera_setting', setting_str,
               '--k', '20',
               '--n_output_imgs', str(img_n)
               ]
    print(' '.join(command))
    subp=subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return 

def psnr_run(p1,p2, show=False):
    commend = f"python validate/code/pic_psnr.py {p1} {p2}"
    subp=subprocess.Popen(commend, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return

def msssim_run(p1,p2, show=False):
    commend = f"python validate/code/pic_mssim.py {p1} {p2}"
    subp=subprocess.Popen(commend, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return 

def lpips_run(p1,p2, show=False):
    commend = f"python validate/code/pic_lpips.py {p1} {p2}"
    subp=subprocess.Popen(commend, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return

if __name__ == "__main__":
    tag = 'tm20'
    gt_name = 'tm20-gt'
    names = ['-01','-03','-06']
    #names = ['-01','-02','-04']
    names = [tag+n for n in names]
    pts_run(gt_name,(60,400,400),'spiral',72,True)
    factor = 512 if tag[:2]=='ld' else 256
    for name in names:
        rescale_run(name,factor,True)
        pts_run(name,(30,200,200),'spiral',144,True)
    for name in names:
        #rescale_run(name,factor,True)
        #pts_run(name,(30,200,200),'spiral',144,True)
        psnr_run(name,gt_name,True)
    rescale_run()