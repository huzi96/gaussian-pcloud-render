import argparse
import sys
import numpy as np
from simple_raw_render import PCML_Render, generate_cam, PointCloud, Simple_Render
from simple_raw_render import get_gt, save_pic
import open3d as o3d
import torch
import subprocess
import time
from structures import Camera

point_light_dict = {
    'longdress': {
        'xyz_w': [
            torch.tensor([5., -5., -5.]),
            torch.tensor([-5., 5., -5.]),
            torch.tensor([0., -5., -5.]),],
        'color': [
            torch.tensor([1., 1., 1.]),
            torch.tensor([1., 1., 1.]),
            torch.tensor([1., 1., 1.])],
        'light_coeff': [0.7, 0.6, 0.3, 0.1]
    },
}


def psnr_run(p1, p2, show=False):
    commend = f"python pic_psnr.py {p1} {p2}"
    subp=subprocess.Popen(commend, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return

def msssim_run(p1, p2, show=False):
    commend = f"python pic_mssim.py {p1} {p2}"
    subp=subprocess.Popen(commend, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return 

def lpips_run(p1, p2, show=False):
    commend = f"python pic_lpips.py {p1} {p2}"
    subp=subprocess.Popen(commend, shell=True, stdout=subprocess.PIPE)
    c=subp.stdout.readline()
    while c:
        if show: print(c)
        c=subp.stdout.readline()
    return

def get_camera_info(args):
    if args.cam_mode == 'udlrfb':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'udlrfb', 'n_imgs':6,
            }
        camera, traj = generate_cam(cam_info, False, True)
    elif args.cam_mode == 'circle':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'circle', 'n_imgs':args.num_frames, 'd': 0, 'r': 3,
            'center_angles': [90, 0], 'alt_yaxis': False
            }
        camera, traj = generate_cam(cam_info, False, True)
    elif args.cam_mode == 'spiral':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'spiral', 'n_imgs':6, 'min_r': 2, 'max_r': 2, 'num_circle': 1, 'r_freq': 1
            }
        camera, traj = generate_cam(cam_info, False, True)
    elif args.cam_mode == 'plot1':
        # Stage 1: circle for 300 frames
        cam_info = {
            'fov':args.fov, 'width_px':1024, 'height_px':1024, 'mode':'circle', 'n_imgs':150, 'd': 0, 'r': 3,
            'center_angles': [90, 0], 'alt_yaxis': False
            }
        cam_stage1, traj_stage1 = generate_cam(cam_info, False, True)
        # Stage 2: zoom in for 30 frames
        r_list = np.linspace(3, 1.5, 30)
        cam_info = {
            'fov':args.fov, 'width_px':1024, 'height_px':1024, 'mode':'circle', 'n_imgs':1, 'd': 0, 'r': 3,
            'center_angles': [90, 0], 'alt_yaxis': False
            }
        cam_stage2, traj_stage2_init = generate_cam(cam_info, False, True)
        for i in range(29):
            cam_info['r'] = r_list[i]
            cam = generate_cam(cam_info, False, False)
            cam_stage2 = Camera.cat([cam_stage2, cam], dim=1)
        
        # Stage 3: stay for 60 frames
        cam_info = {
            'fov':args.fov, 'width_px':1024, 'height_px':1024, 'mode':'circle', 'n_imgs':1, 'd': 0, 'r': 1.5,
            'center_angles': [90, 0], 'alt_yaxis': False
            }
        cam_stage3, _ = generate_cam(cam_info, False, True)
        for i in range(59):
            cam = generate_cam(cam_info, False, False)
            cam_stage3 = Camera.cat([cam_stage3, cam], dim=1)

        # Stage 4: zoom out for 30 frames
        r_list = np.linspace(1.5, 3, 30)
        cam_info = {
            'fov':args.fov, 'width_px':1024, 'height_px':1024, 'mode':'circle', 'n_imgs':1, 'd': 0, 'r': 1.5,
            'center_angles': [90, 0], 'alt_yaxis': False
            }
        cam_stage4, traj_stage4_init = generate_cam(cam_info, False, True)
        for i in range(29):
            cam_info['r'] = r_list[i]
            cam = generate_cam(cam_info, False, False)
            cam_stage4 = Camera.cat([cam_stage4, cam], dim=1)
        
        # Stage 5: stay for 30 frames
        cam_info = {
            'fov':args.fov, 'width_px':1024, 'height_px':1024, 'mode':'circle', 'n_imgs':1, 'd': 0, 'r': 3,
            'center_angles': [90, 0], 'alt_yaxis': False
            }
        cam_stage5, _ = generate_cam(cam_info, False, True)
        for i in range(29):
            cam = generate_cam(cam_info, False, False)
            cam_stage5 = Camera.cat([cam_stage5, cam], dim=1)
        # Concatenate all stages
        camera = Camera.cat([cam_stage1, cam_stage2, cam_stage3, cam_stage4, cam_stage5], dim=1)

    else:
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':args.cam_json, 'n_imgs':12,
        }
        camera, traj = generate_cam(cam_info, False, True)
    
    if args.use_t_indices:
        t_idx = np.arange(0, args.num_frames // 2 - 1, 0.5)
        t_idx = np.round(t_idx).astype(np.int32)
        np.save(args.t_idx_pth, t_idx)
    torch.save(camera.state_dict(), args.cam_save_path)

def get_pcrender_renders(args):
    ckpt = args.ckpt
    torch.backends.cudnn.benchmark = True
    rdr = PCML_Render(ckpt, voxelized=args.voxelized, scale_factor=args.scale_factor, offset=args.offset)
    if args.cam_mode == 'udlrfb':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'udlrfb', 'n_imgs':6,
            }
        camera = generate_cam(cam_info)
    elif args.cam_mode == 'circle':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'circle', 'n_imgs':12, 'd': 0, 'r': 3,
            'center_angles': [90, 0], 'alt_yaxis': False
            }
        camera = generate_cam(cam_info)
    elif args.cam_mode == 'spiral':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'spiral', 'n_imgs':6, 'min_r': 2, 'max_r': 2, 'num_circle': 1, 'r_freq': 1
            }
        camera = generate_cam(cam_info)
    else:
        cam_info = {
            'fov':args.fov, 'width_px':1024, 'height_px':1024, 'mode':args.cam_json, 'n_imgs':12,
        }
        camera = generate_cam(cam_info)

    ids = args.id_list.split(',')
    input_offset = np.array(args.input_offset.split(','), dtype=np.float32)
    print('[Info] input_offset:', input_offset)
    for id in ids:
        print('[Info] Processing', id)
        # Get mesh GT and PCRender GT
        tm20 = f'{args.dataset_root}/{id}/'
        rpth = args.rpth
        if not args.metric_only:
            pcd_pth = tm20 + 'pcd_0.ply'
            o3d_pcd = o3d.io.read_point_cloud(pcd_pth)
            o3d_pts = np.asarray(o3d_pcd.points)
            pts_center = np.mean(o3d_pts, axis=0)
            print('[Info] pts_center:', pts_center)
            if args.down_sample_ratio != 1.0:
                pts = np.asarray(o3d_pcd.points)
                clr = np.asarray(o3d_pcd.colors)
                down_indices = np.random.choice(pts.shape[0], int(pts.shape[0] * args.down_sample_ratio), replace=False)
                pts = pts[down_indices]
                clr = clr[down_indices]
                o3d_pcd.points = o3d.utility.Vector3dVector(pts)
                o3d_pcd.colors = o3d.utility.Vector3dVector(clr)
            gt_pcd = PointCloud.from_o3d_pcd(o3d_pcd)
                
            if not args.skip_mesh:
                mesh_gt = get_gt(tm20 + f'{id}.obj', camera)
                mesh_gt_rgb = mesh_gt['ray_rgbs']
                mesh_gt_rgb = mesh_gt_rgb + (1 - mesh_gt['hit_map'].unsqueeze(-1)) * torch.from_numpy(args.background_color)

                mesh_gt_normal = mesh_gt['surface_normals_w']
                save_pic(
                    mesh_gt_rgb, rpth + f'{id}_mesh_gt', 'rgb')
                save_pic(
                    mesh_gt_normal, rpth + f'{id}_mesh_gt', 'normal_w',
                    hit_map=mesh_gt['hit_map'].unsqueeze(-1))
            
            with torch.no_grad():
                pcrender_dict = rdr.render(
                    gt_pcd, scale=None , cam=camera,
                    fov=cam_info['fov'],
                    enable_opacity=True,
                    super_sample_rate=args.pcrender_ssrate,
                    input_offset=input_offset,
                    point_light=point_light_dict.get(id, None),
                    est_normal_from_ellipsoid=False,
                    background_color=torch.from_numpy(args.background_color).float().to(rdr.device)
                    )
                pcrender_result = pcrender_dict['rgb']
        
            save_pic(pcrender_result, rpth + f'{id}_pcrender', type='rgb')
            if pcrender_dict['normal'] is not None:
                save_pic(pcrender_dict['normal'] , rpth + f'{id}_pcrender', type='normal_w')
            if pcrender_dict['xyz_w'] is not None:
                save_pic(pcrender_dict['xyz_w'] , rpth + f'{id}_pcrender', type='xyz_w')
            if pcrender_dict.get('shaded', None) is not None:
                save_pic(pcrender_dict['shaded'] , rpth + f'{id}_pcrender', type='shaded')

        if not args.skip_mesh:
            psnr_run(rpth + f'{id}_pcrender', rpth + f'{id}_mesh_gt', show=True)
            msssim_run(rpth + f'{id}_pcrender', rpth + f'{id}_mesh_gt', show=True)
            lpips_run(rpth + f'{id}_pcrender', rpth + f'{id}_mesh_gt', show=True)

def get_simple_renders(args):
    rdr = Simple_Render(voxelized=args.voxelized, scale_factor=args.scale_factor, offset=args.offset)
    if args.cam_mode == 'udlrfb':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'udlrfb', 'n_imgs':6,
            }
        camera = generate_cam(cam_info)
    elif args.cam_mode == 'circle':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'circle', 'n_imgs':12, 'd': 0, 'r': 3,
            'center_angles': [90, 0], 'alt_yaxis': False
            }
        camera = generate_cam(cam_info)
    elif args.cam_mode == 'spiral':
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':'spiral', 'n_imgs':6, 'min_r': 2, 'max_r': 2, 'num_circle': 1, 'r_freq': 1
            }
        camera = generate_cam(cam_info)
    else:
        cam_info = {
            'fov':args.fov, 'width_px':512, 'height_px':512, 'mode':args.cam_json, 'n_imgs':12,
        }
        camera = generate_cam(cam_info)

    ids = args.id_list.split(',')
    input_offset = np.array(args.input_offset.split(','), dtype=np.float32)
    print('[Info] input_offset:', input_offset)
    for id in ids:
        print('[Info] Processing', id)
        # Get mesh GT and PCRender GT
        tm20 = f'{args.dataset_root}/{id}/'
        rpth = args.rpth
        if not args.metric_only:
            pcd_pth = tm20 + 'pcd_0.ply'
            o3d_pcd = o3d.io.read_point_cloud(pcd_pth)
            # Downsample
            if args.down_sample_ratio != 1.0:
                o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=2)

            # find average point distance
            kd_tree = o3d.geometry.KDTreeFlann(o3d_pcd)
            pts = np.asarray(o3d_pcd.points)
            dist_sum = 0
            for i in range(pts.shape[0]):
                _, indices, _ = kd_tree.search_knn_vector_3d(pts[i], 2)
                dist_sum += np.linalg.norm(pts[i] - pts[indices[1]])
            avg_dist = dist_sum / pts.shape[0]
            print('[Info] avg_dist:', avg_dist)

            st_time = time.time()
            # estimate normals
            o3d_pcd.estimate_normals()
            # o3d_pcd.orient_normals_consistent_tangent_plane(100)
            print('[Info] estimate_normals time:', time.time() - st_time)

            gt_pcd = PointCloud.from_o3d_pcd(o3d_pcd)
            if not args.skip_mesh:
                mesh_gt = get_gt(tm20 + f'{id}.obj', camera)
                mesh_gt_rgb = mesh_gt['ray_rgbs']
                if args.background_color != 0:
                    mesh_gt_rgb = mesh_gt_rgb + (1 - mesh_gt['hit_map'].unsqueeze(-1)) * args.background_color

                mesh_gt_normal = mesh_gt['surface_normals_w']
                save_pic(mesh_gt_rgb, rpth + f'{id}_mesh_gt', 'rgb')
                save_pic(mesh_gt_normal, rpth + f'{id}_mesh_gt', 'normal_w', hit_map=mesh_gt['hit_map'].unsqueeze(-1))
        
            with torch.no_grad():
                pcrender_gt_dict = rdr.render(
                    gt_pcd, scale=None , cam=camera,
                    fov=cam_info['fov'],
                    enable_opacity=False,
                    super_sample_rate=args.pcrender_ssrate,
                    input_offset=input_offset,
                    point_light=point_light_dict.get(id, None),
                    est_normal_from_ellipsoid=False,
                    background_color=args.background_color,
                    sigma=args.sigma
                    )
                pcrender_gt = pcrender_gt_dict['rgb']
        
            save_pic(pcrender_gt, rpth + f'{id}_simple_sigma_{args.sigma}', type='rgb')
            if pcrender_gt_dict['normal'] is not None:
                save_pic(pcrender_gt_dict['normal'] , rpth + f'{id}_simple_sigma_{args.sigma}', type='normal_w')
            if pcrender_gt_dict['xyz_w'] is not None:
                save_pic(pcrender_gt_dict['xyz_w'] , rpth + f'{id}_simple_sigma_{args.sigma}', type='xyz_w')
            if pcrender_gt_dict.get('shaded', None) is not None:
                save_pic(pcrender_gt_dict['shaded'] , rpth + f'{id}_simple_sigma_{args.sigma}', type='shaded')

        if not args.skip_mesh:
            psnr_run(rpth + f'{id}_simple_sigma_{args.sigma}', rpth + f'{id}_mesh_gt', show=True)
            msssim_run(rpth + f'{id}_simple_sigma_{args.sigma}', rpth + f'{id}_mesh_gt', show=True)
            lpips_run(rpth + f'{id}_simple_sigma_{args.sigma}', rpth + f'{id}_mesh_gt', show=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['pcrender', 'simple', 'cam'])
    parser.add_argument('--ckpt', type=str, default='./models/1-21-2/train/checkpoint/model_epoch39.pth')
    parser.add_argument('--id_list', type=str, default='0519')
    parser.add_argument('--dataset_root', type=str, default='./example/THuman-256')
    parser.add_argument('--rpth', type=str, default='validate/res/render/')
    parser.add_argument('--pcrender_ssrate', type=int, default=2, help='super sample rate of pcrender')
    parser.add_argument('--skip_mesh', action='store_true', help='whether to skip mesh gt')
    parser.add_argument('--fov', type=int, default=45)
    parser.add_argument('--voxelized', action='store_true', help='whether to use voxelized pcd')
    parser.add_argument('--scale_factor', type=int, default=256, help='scale factor for voxelized pcd')
    parser.add_argument('--input_offset', type=str, default='0,0,0')
    parser.add_argument('--cam_mode', type=str, default='circle')
    parser.add_argument('--cam_json', type=str, default='')
    parser.add_argument('--background_color', type=str, default='1')
    parser.add_argument('--metric_only', action='store_true', help='whether to only compute metric')
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--simple_on', action='store_true', help='whether to use simple render')
    parser.add_argument('--offset', type=int, default=512)
    parser.add_argument('--cam_save_path', type=str, default='validate/res/cam/cam.pt')
    parser.add_argument('--down_sample_ratio', type=float, default=1.0)
    args = parser.parse_args(sys.argv[1:])
    background_color = args.background_color.split(',')
    if len(background_color) == 1:
        args.background_color = np.array(
            [float(background_color[0]), float(background_color[0]), float(background_color[0])])
    else:
        args.background_color = np.array(background_color, dtype=np.float32) / 255.
    if args.task == 'pcrender':
        get_pcrender_renders(args)
    elif args.task == 'simple':
        get_simple_renders(args)
    elif args.task == 'cam':
        get_camera_info(args)

    