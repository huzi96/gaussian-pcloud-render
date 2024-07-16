#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#


import argparse
import typing as T
from timeit import default_timer as timer

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from base_train import BaseTrainProcess
from pointersect_utils.print_and_save import Logger
import dataset_helper
import mesh_dataset_v2
import structures
import os
import copy
from models.model_v2 import PCEncoder

from torch.nn.utils.rnn import pad_sequence
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from models.sh_utils import *
import math
import lpips
from models.loss_utils import ssim

def get_neighbor_topk(
        vertices,
        vertices2,
        neighbor_num: int):
        # find k neigh points in v2 for v1
        # Return: (bs, vertice_num, neighbor_num)
        bs, v, _ = vertices.size()
        device = vertices.device
        inner = torch.bmm(vertices, vertices2.transpose(1, 2)) #(bs, v1, v2)
        quadratic1 = torch.sum(vertices**2, dim= 2) #(bs, v1)
        quadratic2 = torch.sum(vertices2**2, dim= 2) #(bs, v2)
        distance = inner * (-2) + quadratic2.unsqueeze(1) + quadratic1.unsqueeze(2)#(bs,v1,v2)
        neighbor_tk = torch.topk(distance, k= neighbor_num, dim= -1, largest= False)
        return neighbor_tk

def pcgc_rescale(input_xyz, offset=512, factor=256, device=None):
    pnum = input_xyz.shape[0]
    res = (input_xyz - offset) / factor
    device = input_xyz.device
    return res.to(device)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

BT601_RGB2YUV_MAT = np.array(
    [[0.299, 0.587, 0.114],
     [-0.168736, -0.331264, 0.5],
     [0.5, -0.418688, -0.081312]])

BT601_YUV2RGB_MAT = np.array(
    [[1.0, 0.0, 1.402],
     [1.0, -0.344136, -0.714136],
     [1.0, 1.772, 0.0]])

class RGB2YUV(torch.nn.Module):
    """RGB to YUV conversion module.
    
    Assume input is torch Tensor in range [0, 255]."""
    def __init__(self):
        super(RGB2YUV, self).__init__()
        rgb2yuv_mat = torch.from_numpy(BT601_RGB2YUV_MAT).float()
        uv_offset = torch.tensor([0, 0.5, 0.5]).float()
        self.register_buffer('rgb2yuv_mat', rgb2yuv_mat)
        self.register_buffer('uv_offset', uv_offset)
    
    def forward(self, x):
        '''Convert RGB image to YUV image.'''
        yuv = torch.einsum('bchw,cd->bdhw', x, self.rgb2yuv_mat.t())
        return yuv + self.uv_offset.view(1, 3, 1, 1)
    
class YUV2RGB(torch.nn.Module):
    """YUV to RGB conversion module.
    
    Assume input is torch Tensor in range [0, 255]."""
    def __init__(self):
        super(YUV2RGB, self).__init__()
        yuv2rgb_mat = torch.from_numpy(BT601_YUV2RGB_MAT).float()
        uv_offset = torch.tensor([0, 0.5, 0.5]).float()
        self.register_buffer('yuv2rgb_mat', yuv2rgb_mat)
        self.register_buffer('uv_offset', uv_offset)
    
    def forward(self, x):
        '''Convert YUV image to RGB image.'''
        rgb = torch.einsum('bchw,cd->bdhw', x - self.uv_offset.view(1, 3, 1, 1), self.yuv2rgb_mat.t())
        return rgb

class TrainPointersectProcess(BaseTrainProcess):

    def __init__(
            self,
            ## dataset_info
            dataset_name: str = 'tex',  # name of the dataset to train on, determines which dataset to download
            dataset_root_dir: str = 'datasets/tex-models',  # where the meshes are
            mesh_filename: T.Union[str, T.List[str]] = 'bunny.obj',
            test_mesh_filename: str = 'cat.obj',
            batch_size: int = 2,
            n_target_imgs: int = 2,
            n_imgs: int = 3,
            width_px: int = 200,
            height_px: int = 200,
            target_width_px: int = 20,
            target_height_px: int = 20,
            fov: int = 60.,
            max_angle: float = 30.,
            local_max_angle: float = 3.,
            max_translate_ratio: float = 2.0,  # not used
            ray_perturbation_angle: float = 3,  # not used
            total: int = 10000,
            pcd_subsample: int = 1,  # not used, replaced by min_subsample
            dataset_rng_seed: int = 0,
            k: int = 40,
            randomize_translate: bool = False,
            # not used  # whether translation amount is randomized, see utils.rectify_points
            ray_radius: float = 0.1,  # radius of the ray, used in pr
            num_threads: int = 0,
            train_cam_path_mode: str = 'random',  # not used  # random/circle # support different camera trajectory
            generate_point_cloud_input: bool = False,  # not used
            clean_mesh: bool = True,  # not used  # if true, clean the obj file
            cleaned_root_dir: str = 'datasets/cleaned_models',  # not used  # where the cleaned obj meshes are saved
            skip_existed_cleaned_mesh: bool = False,  # if true, will not clean the obj file again if existed
            render_method: str = 'ray_cast',  # 'ray_cast', 'rasterization'
            min_subsample: int = 1,
            max_subsample: int = 1,  # None: same as min_subsample
            min_k_ratio: float = 1.,
            max_k_ratio: float = 1.,  # None: same as max_k_ratio
            mesh_scale: float = 1.,
            min_r: float = 0.5,
            max_r: float = 3.,
            rand_r: float = 0.,
            texture_mode: str = 'ori',  # 'files', 'imagenet'
            texture_crop_method: T.Union[int, str] = 'ori',  # or an int p indiciating the min p * p crop
            texture_filenames: T.List[str] = None,
            use_bucket_sampler: bool = True,
            mix_meshes: bool = False,
            min_num_mesh: int = 1,
            max_num_mesh: int = 2,
            radius_scale: float = 2.,
            total_combined: int = None,
            ## model_info
            learn_dist: bool = False,
            num_layers: int = 4,  # 4,  # 3,
            dim_feature: int = 512,  # 256,
            num_heads: int = 4,
            encoding_type: str = 'pos',  # pos/ siren  # support different ways of position encoding
            positional_encoding_num_functions: int = 10,  # to turn off position encoding, set to 0
            positional_encoding_include_input: bool = True,
            positional_encoding_log_sampling: bool = True,
            nonlinearity: str = 'silu',
            dim_mlp: int = 512,  # 1024,  # 512,
            dropout: float = 0.1,
            direction_param: str = 'norm_vec',
            estimate_surface_normal_weights: bool = False,
            estimate_image_rendering_weights: bool = True,
            use_rgb_as_input: bool = False,
            use_dist_as_input: bool = False,  # if true, use |x|,|y|,|z| and sqrt(x^2+y^2) in ray space as input
            use_zdir_as_input: bool = False,  # if true, use camera viewing direction (2 vector, 3 dim) as input
            use_dps_as_input: bool = False,  # if true, use local frame width (1 value, 1 dim) as input
            use_dpsuv_as_input: bool = False,  # if true, use local frame (2 vectors, 6 dim) as input
            use_layer_norm: bool = False,  # if true, enable layer norm
            use_pr: bool = False,  # if true, use pr to find neighbor points within a fixed distance to ray
            use_additional_invalid_token: bool = False,  # if true, an extra invalid token will be used in transformer
            dim_input_layers: T.List[int] = None,
            use_vdir_as_input: bool = False,
            use_rgb_indicator: bool = False,  # whether to add a binary indicator saying input has valid rgb
            use_feature_indicator: bool = False,  # whether to add a binary indicator saying input has valid feature
            ## optim_info
            optim_method: str = 'adam',  # 'adam_tf'
            learning_rate: float = 1.0e-4,
            lr_factor: float = 0.1,
            num_warmup_steps: int = 4000,
            max_grad_val: float = 1.0,
            use_amp: bool = False,
            loss_weight_t: float = 10.,
            loss_weight_t_l1: float = 0.,
            loss_weight_normal: float = 1.,
            loss_weight_normal_l1: float = 0.,
            loss_weight_plane_normal: float = 1.,
            loss_weight_plane_normal_l1: float = 0.,
            loss_weight_hit: float = 1.0,
            loss_weight_rgb: float = 1.0,
            loss_weight_rgb_normal: float = 0,
            loss_weight_rgb_normal_dot: float = 0,
            loss_weight_rgb_normal_dot_l1: float = 0,
            loss_rgb_type: str = 'l1',  # 'l2'
            focal_loss_gamma: float = 2.0,
            focal_loss_alpha: float = 0.5,
            learn_ray_rgb: bool = True,
            random_drop_rgb_rate: float = 0,  # probability that the rgb will be randomly dropped
            random_drop_sample_feature_rate: float = 0,  # probability that zdir, dps, dpsuv will be randomly dropped
            pcd_noise_std: float = 0,  # std of the gaussian noise added to the input point cloud
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # set default values
        self.dataset_info = dict(
            dataset_name=dataset_name,
            dataset_root_dir=dataset_root_dir,
            mesh_filename=mesh_filename,
            test_mesh_filename=test_mesh_filename,
            batch_size=batch_size,
            n_target_imgs=n_target_imgs,
            n_imgs=n_imgs,
            width_px=width_px,
            height_px=height_px,
            target_width_px=target_width_px,
            target_height_px=target_height_px,
            fov=fov,
            max_angle=max_angle,
            local_max_angle=local_max_angle,
            max_translate_ratio=max_translate_ratio,
            ray_perturbation_angle=ray_perturbation_angle,
            total=total,
            pcd_subsample=pcd_subsample,
            dataset_rng_seed=dataset_rng_seed,
            k=k,
            randomize_translate=randomize_translate,
            ray_radius=ray_radius,
            num_threads=num_threads,
            train_cam_path_mode=train_cam_path_mode,
            generate_point_cloud_input=generate_point_cloud_input,
            clean_mesh=clean_mesh,
            cleaned_root_dir=cleaned_root_dir,
            skip_existed_cleaned_mesh=skip_existed_cleaned_mesh,
            render_method=render_method,
            max_subsample=max_subsample if max_subsample is not None else min_subsample,
            min_subsample=min_subsample,
            min_k_ratio=min_k_ratio,
            max_k_ratio=max_k_ratio if max_k_ratio is not None else min_k_ratio + 1e-6,
            mesh_scale=mesh_scale,
            rand_r=rand_r,
            min_r=min_r,
            max_r=max_r,
            texture_mode=texture_mode,
            texture_crop_method=texture_crop_method,
            texture_filenames=texture_filenames,
            use_bucket_sampler=use_bucket_sampler,
            mix_meshes=mix_meshes,
            min_num_mesh=min_num_mesh,
            max_num_mesh=max_num_mesh,
            radius_scale=radius_scale,
            total_combined=total_combined)
        self.pcml_info = dict()

        if self.dataset_info.get('max_subsample', 1) != self.dataset_info.get('min_subsample', 1):
            assert self.dataset_info['batch_size'] == 1

        self.model_info = dict(
            learn_dist=learn_dist,
            num_layers=num_layers,
            dim_feature=dim_feature,
            num_heads=num_heads,
            encoding_type=encoding_type,
            positional_encoding_num_functions=positional_encoding_num_functions,
            positional_encoding_include_input=positional_encoding_include_input,
            positional_encoding_log_sampling=positional_encoding_log_sampling,
            nonlinearity=nonlinearity,
            dim_mlp=dim_mlp,
            dropout=dropout,
            direction_param=direction_param,
            estimate_surface_normal_weights=estimate_surface_normal_weights,
            estimate_image_rendering_weights=estimate_image_rendering_weights,
            use_rgb_as_input=use_rgb_as_input,
            use_dist_as_input=use_dist_as_input,
            use_zdir_as_input=use_zdir_as_input,
            use_dps_as_input=use_dps_as_input,
            use_dpsuv_as_input=use_dpsuv_as_input,
            use_layer_norm=use_layer_norm,
            use_pr=use_pr,
            use_additional_invalid_token=use_additional_invalid_token,
            dim_input_layers=dim_input_layers,
            use_vdir_as_input=use_vdir_as_input,
            use_rgb_indicator=use_rgb_indicator,
            use_feature_indicator=use_feature_indicator,
        )
        if self.model_info['use_rgb_indicator']:
            assert self.model_info['use_rgb_as_input']
        if self.model_info['use_feature_indicator']:
            assert self.model_info['use_dist_as_input'] or self.model_info['use_zdir_as_input'] or \
                   self.model_info['use_dps_as_input'] or self.model_info['use_dpsuv_as_input'] or \
                   self.model_info['use_vdir_as_input']
        

        self.optim_info = dict(
            optim_method=optim_method,
            learning_rate=learning_rate,
            lr_factor=lr_factor,
            num_warmup_steps=num_warmup_steps,
            use_amp=use_amp,
            max_grad_val=max_grad_val,
            loss_weight_t=loss_weight_t,
            loss_weight_t_l1=loss_weight_t_l1,
            loss_weight_normal=loss_weight_normal,
            loss_weight_normal_l1=loss_weight_normal_l1,
            loss_weight_hit=loss_weight_hit,
            loss_weight_rgb=loss_weight_rgb,
            loss_weight_rgb_normal=loss_weight_rgb_normal,
            loss_weight_rgb_normal_dot=loss_weight_rgb_normal_dot,
            loss_weight_rgb_normal_dot_l1=loss_weight_rgb_normal_dot_l1,
            loss_weight_plane_normal=loss_weight_plane_normal,
            loss_weight_plane_normal_l1=loss_weight_plane_normal_l1,
            loss_rgb_type=loss_rgb_type,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
            learn_ray_rgb=learn_ray_rgb,
            random_drop_rgb_rate=random_drop_rgb_rate,
            random_drop_sample_feature_rate=random_drop_sample_feature_rate,
            pcd_noise_std=pcd_noise_std,
        )
        
        
        self.model: T.Union[torch.nn.Module, None] = None  # this is the model
        self.pcml_model = None
        self.losses_name = set()
        self.outputs_name = set()
        self.global_step = 0
        self.nan_count = 0

        self._register_var_to_save(
            [
                "dataset_info", "model_info", "optim_info", "losses_name", "outputs_name",
            ])  # dataset_info can change when resuming, so only save for future reference

        self._register_var_to_load([])  # these settings should not change when resuming, so reload
        # load options
        self.load_options(filename=self.process_info['config_filename'])
        self.batch_size = self.dataset_info['batch_size']

        print("weight lamda:",self.optim_info["weight_lamda"])

        self._register_output(
            [
                "pointersect_record",
            ])

        self._register_loss(
            [
                "loss_t",
                "loss_t_l1",
                "loss_normal",
                "loss_normal_l1",
                "loss_normal_l2",
                "loss_normal_reg",
                "loss_plane_normal",
                "loss_plane_normal_l1",
                "loss_total",
                "loss_cd",
                "loss_hit",
                "loss_rgb",
                'loss_rgb_normal',
                'loss_rgb_normal_dot',
                'loss_rgb_normal_dot_l1',
                "rmse_theta",
                "rmse_theta_l1",
                "rmse_t",
                "rmse_plane_theta",
                "rmse_plane_theta_l1",
                "loss_pcgc_cls",
                "loss_dc",
                "loss_lpips",
                'loss_bpp_clr',
                'loss_bpp_hyper',
                'loss_point_normal_l1',
                'loss_ssim'
            ])


    def get_dataloaders(self):
        """This function is called after setup_assets."""

        self.logger.info(f'Creating datasets...')

        # camera settings
        input_camera_setting = dict(
            width_px=self.dataset_info['width_px'],
            height_px=self.dataset_info['height_px'],
            fov=self.dataset_info['fov'],
        )
        # we want input point cloud covers the entire mesh (with various sampling rate)
        input_camera_trajectory_params = dict(
            mode='random',
            min_r=self.dataset_info.get('mesh_scale', 1.),
            max_r=self.dataset_info.get('mesh_scale', 1.) * 3,
            max_angle=self.dataset_info['max_angle'],
            rand_r=self.dataset_info['rand_r'],
            local_max_angle=self.dataset_info['local_max_angle'],
            r_freq=1,
            max_translate_ratio=self.dataset_info['max_translate_ratio'],
        )
        output_camera_setting = dict(
            width_px=self.dataset_info['target_width_px'],
            height_px=self.dataset_info['target_height_px'],
            fov=self.dataset_info['fov'],
        )
        output_camera_trajectory_params = dict(
            mode=self.dataset_info.get('output_cam_mode', 'random'),
            min_r=self.dataset_info.get('min_r', 1),
            max_r=self.dataset_info.get('max_r', 3),
            max_angle=self.dataset_info['max_angle'],
            rand_r=self.dataset_info['rand_r'],
            local_max_angle=self.dataset_info['local_max_angle'],
            r_freq=1,
            max_translate_ratio=self.dataset_info['max_translate_ratio'],
        )
        
        dataset_dict = dataset_helper.get_dataset(
            dataset_name=self.dataset_info['dataset_name'],
            dataset_info=self.dataset_info,
            input_camera_setting=input_camera_setting,
            input_camera_trajectory_params=input_camera_trajectory_params,
            output_camera_setting=output_camera_setting,
            output_camera_trajectory_params=output_camera_trajectory_params,
            rank=self.process_info['rank'],
            world_size=self.process_info['global_world_size'],
            printout=self.process_info['rank'] == 0,
            dataset_root_dir=self.dataset_info['dataset_root_dir'],
            num_pcd=self.dataset_info.get('num_pcd', 1),
            mode=self.dataset_info.get('sample_mode', 'uniform'),
        )
        dataset: mesh_dataset_v2.MeshConcatDataset = dataset_dict['dataset']
        val_dataset: mesh_dataset_v2.MeshConcatDataset = dataset_dict['val_dataset']
        test_dataset: mesh_dataset_v2.MeshConcatDataset = dataset_dict['test_dataset']

        # get dataloader
        self.logger.info('Creating dataloaders...')

        collate_fn = mesh_dataset_v2.MeshDatasetCollate()

        # add sampler to support distributed run
    
        self.train_sampler = None
        self.valid_sampler = None
        if test_dataset is not None:
            self.test_sampler = None

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.dataset_info['batch_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.dataset_info['num_threads'],
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.dataset_info['batch_size'],
            shuffle=(self.valid_sampler is None),
            num_workers=self.dataset_info['num_threads'],
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=self.valid_sampler,
            drop_last=True,
        )
        if test_dataset is None:
            test_dataloader = None
        else:
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.dataset_info['batch_size'],
                shuffle=(self.test_sampler is None),
                num_workers=self.dataset_info['num_threads'],
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=self.test_sampler,
                drop_last=True,
            )

        return dataloader, val_dataloader, test_dataloader
    
    def get_pcml_config(self):
        color_info = copy.deepcopy(self.pcml_info)
        coor_info = self.pcml_info
        for key in color_info.keys():
            if key[:5] == 'color':
                color_info[key[6:]] = color_info[key]
        coor_info['is_color'] = False
        color_info['is_color'] = True
        return coor_info,color_info

    def construct_models(self):
        self.pcml_model =  PCEncoder(self.pcml_info).cuda()
        self.randomize_background = self.pcml_info['randomize_background']
        self.pruning = ME.MinkowskiPruning()            

        load_coor = self.pcml_info.get('load_pcml_ckpt', False)
        if load_coor:
            pcml_ckpt_path = self.pcml_info["pcml_ckpt_pth"]
            self.pcml_ckpt = torch.load(pcml_ckpt_path)
            ret = self.pcml_model.load_state_dict(self.pcml_ckpt["pcml_model"],strict=False)
            print(ret)
            print('Loaded weights for pcml.')
        else:
            self.pcml_ckpt = None
                
        self.rgb_to_yuv = RGB2YUV().cuda()
        self.yuv_to_rgb = YUV2RGB().cuda()

    def construct_optimizers(self):

        self.update_paras = list(self.pcml_model.parameters())

        self.optimizer = torch.optim.Adam(self.update_paras, lr=self.optim_info['learning_rate'])
        self.scaler = torch.cuda.amp.GradScaler()

        if self.optim_info["load_optimizer"] and self.pcml_ckpt is not None:
            self.optimizer.load_state_dict(self.pcml_ckpt["optimizer"])
            self.scaler.load_state_dict(self.pcml_ckpt["scaler"])

        self.loss_l1 = torch.nn.L1Loss(reduction='none')
        self.loss_mse = torch.nn.MSELoss(reduction='none')
        if self.optim_info['loss_weight_lpips'] > 0:
            self.lpips = lpips.LPIPS(net='vgg').to(self.device)

    def get_gt_color(
            self,
            upsampled_coor,
            gt_coor,
            gt_color,
    ):
    #warning: select the nearest point
        topk = get_neighbor_topk(upsampled_coor,gt_coor,self.optim_info['color_gt_neighbor'])
        dis,idx = topk[0].unsqueeze(-1).detach(),topk[1] #bs,v,nei,1
        id0 = torch.arange(self.batch_size).view(-1,1,1)
        neigh_color = gt_color[id0,idx] # bs,v,nei,3
        upsampled_color = neigh_color.squeeze(-2)
        alpha_glue = 1.0/torch.mean(dis)
        nei_weight = torch.exp( -1 * alpha_glue * dis)
        nei_weight_norm = nei_weight / (torch.sum(nei_weight,dim=-2,keepdim=True)+0.0000001)
        upsampled_color = torch.sum(neigh_color*nei_weight_norm,dim=-2)  #bs,v,3
        return upsampled_color

    def get_rasterize_param_from_camera(
            self,
            camera: structures.Camera, fovX_deg=60., fovY_deg=60.,
            sh_degree=0, bg=None, super_sample_ratio=1,
            manual_width=None, manual_height=None):
        # view_mat = getWorld2View2(np.eye(3), np.ones(3) * 5., scale=1.0)
        world_view_transform = camera.get_H_w2c().to(device=self.device).transpose(-2, -1)
        projection_matrix = getProjectionMatrix(
            znear=0.01, zfar=100,
            fovX=np.pi * fovX_deg / 180, fovY=np.pi * fovY_deg / 180,
            )
        # import IPython
        #IPython.embed()
        projection_matrix = projection_matrix.transpose(-2, -1).cuda()
        mat_shape = world_view_transform.shape
        world_view_transform = world_view_transform.reshape(-1, 4, 4)
        projection_matrix = projection_matrix.unsqueeze(0).expand(world_view_transform.shape[0], -1, -1).reshape(-1, 4, 4)
        full_proj_transform = (
            world_view_transform.bmm(projection_matrix))
        if bg is None:
            bg = torch.zeros((3,)).to(device=self.device)
        else:
            bg = bg.to(device=self.device)
        if manual_width is None or manual_height is None:
            width = camera.width_px * super_sample_ratio
            height = camera.height_px * super_sample_ratio
        else:
            width = manual_width * super_sample_ratio
            height = manual_height * super_sample_ratio
        rasterize_setting = GaussianRasterizationSettings(
            image_height = height,
            image_width = width,
            tanfovx = math.tan(fovX_deg / 180. * math.pi),
            tanfovy = math.tan(fovY_deg / 180. * math.pi),
            bg = bg,
            scale_modifier = 1.0,
            viewmatrix = world_view_transform,
            projmatrix = full_proj_transform,
            sh_degree = sh_degree,
            campos = torch.matmul(camera.H_c2w, torch.tensor([0, 0, 0, 1]).to(device=self.device).float())[..., 0:3],
            prefiltered = False,
            debug = False
        )
        return rasterize_setting

    def augment_color(self, pcd_color, img_color, scale=0.2):
        assert len(pcd_color.shape) == 3 # b, N, c
        assert pcd_color.shape[-1] == 3
        assert len(img_color.shape) == 5 # b, q, h, w, c
        assert img_color.shape[-1] == 3
        assert torch.max(pcd_color) <= 1 and torch.min(pcd_color) >= 0
        # Convert color to YUV
        flatten_pcd_color = pcd_color.reshape(-1, 3, 1, 1)
        flatten_img_color = img_color.reshape(-1, 3, 1, 1)
        flatten_pcd_color_yuv = self.rgb_to_yuv(flatten_pcd_color)
        flatten_img_color_yuv = self.rgb_to_yuv(flatten_img_color)
        pcd_color_yuv = flatten_pcd_color_yuv.reshape(pcd_color.shape)
        img_color_yuv = flatten_img_color_yuv.reshape(img_color.shape)
        # Augment YUV
        b = pcd_color_yuv.shape[0]
        random_perturb = (
            torch.rand(
                (b, 2), device=self.device, dtype=torch.float32
                ) * 2 - 1
                ) * scale
        perturb = torch.cat(
            (torch.zeros((b, 1), device=self.device, dtype=torch.float32), random_perturb), dim=-1
        )
        pcd_color_yuv_aug = pcd_color_yuv + perturb.unsqueeze(-2)
        img_color_yuv_aug = img_color_yuv + perturb.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)

        pcd_color_yuv_aug = torch.clamp(pcd_color_yuv_aug, 0, 1)
        img_color_yuv_aug = torch.clamp(img_color_yuv_aug, 0, 1)

        # Convert back to RGB
        flatten_pcd_color_yuv_aug = pcd_color_yuv_aug.reshape(-1, 3, 1, 1)
        flatten_img_color_yuv_aug = img_color_yuv_aug.reshape(-1, 3, 1, 1)

        flatten_pcd_color_aug = self.yuv_to_rgb(flatten_pcd_color_yuv_aug)
        flatten_img_color_aug = self.yuv_to_rgb(flatten_img_color_yuv_aug)

        pcd_color_aug = flatten_pcd_color_aug.reshape(pcd_color.shape)
        img_color_aug = flatten_img_color_aug.reshape(img_color.shape)
        return pcd_color_aug, img_color_aug

    def _rasterize(self, batchsize,
                   ray_space_points_sparse_recon, primitive_dict,
                   camera_chunks, num_q, bg, h, w,
                   normalize_camera_normal=False):
        decoded_o = primitive_dict['decoded_o']
        decoded_s = primitive_dict['decoded_s']
        decoded_r = primitive_dict['decoded_r']
        decoded_sh = primitive_dict['decoded_sh']
        colors_precomp = primitive_dict['colors_precomp']
        rendered_recons = []

        for i in range(batchsize):
            means3D = ray_space_points_sparse_recon[i]
            means2D = torch.zeros_like(
                means3D, dtype=torch.float32, requires_grad=True, device="cuda"
                ) + 0
            # opacity = torch.zeros_like(means3D[..., 0:1], dtype=torch.float32, requires_grad=True, device="cuda") + 1
            if self.pcml_info.get('enable_opacity',True):
                opacity = decoded_o[i]
            else:
                opacity = torch.ones_like(decoded_o[i])
            radius = np.sqrt(3) / self.pcml_info['scale_factor'] * 6
            scales = decoded_s[i] * radius
            rotations = decoded_r[i]
            cov3D_precomp = None
            if decoded_sh is not None:
                decoded_shs = decoded_sh[i]
                colors_precomp_i = None
            else:
                decoded_shs = None
                colors_precomp_i = colors_precomp[i]
            
            camera_chunk_j = camera_chunks[i].chunk(num_q, dim=1)
            for j in range(num_q):
                raster_settings = self.get_rasterize_param_from_camera(
                    camera_chunk_j[j], bg=bg, sh_degree=self.pcml_info['sh_deg'])
                rasterizer = GaussianRasterizer(raster_settings)
                if normalize_camera_normal:
                    camera_orig = camera_chunk_j[j].get_camera_origin_w()
                    camera_dir = means3D - camera_orig
                    normal_dir_sgn = (torch.sum(camera_dir * colors_precomp_i, -1, keepdim=True) > 0).float() * 2 - 1
                    colors_precomp_i = colors_precomp_i * (-1) * normal_dir_sgn[0]
                rendered_image, radii = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = decoded_shs,
                    colors_precomp = colors_precomp_i,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                rendered_recons.append(rendered_image)
        rendered_recons = torch.stack(rendered_recons, dim=0)
        # radii_recons = torch.stack(radii_recons, dim=0)
        rendered_recons = rendered_recons.reshape(batchsize, num_q, 3, h, w)
        rendered_recons = rendered_recons.permute(0, 1, 3, 4, 2)
        # radii_recons = radii_recons.reshape(batchsize, num_q, 1, gt_ray_rgb.shape[2], gt_ray_rgb.shape[3])
        return rendered_recons

    def _step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
            update: bool,
    ):
        """
        Args:
            epoch:
            bidx:
            batch:
                input_rgbd_images:
                    RGBDImage, (b, q, h, w),  images along q should be used to create point cloud
                ray:
                    Ray, (b, q=n_target_img, ho, wo)  target rays
                ray_gt_dict:
                    ray_rgbs: (b, q=n_target_img, ho, wo, 3)
                    ray_ts: (b, q=n_target_img, ho, wo)
                    surface_normals_w: (b, q=n_target_img, ho, wo, 3)
                    hit_map: (b, q=n_target_img, ho, wo)  1 if hit a surface, 0 otherwise
            update:

        Returns:

        """

        

        with torch.autocast(
                device_type='cuda' if self.process_info['n_gpus'] > 0 else 'cpu',
                enabled=self.optim_info['use_amp'],
        ):
            ray: structures.Ray = batch['ray'].to(device=self.device)  # (b, qo, ho, wo)
            ray_gt_dict = batch['ray_gt_dict']
            gt_ray_rgb: torch.Tensor = ray_gt_dict['ray_rgbs'].to(device=self.device)  # (b, qo, ho, wo, 3)
            gt_ray_ts: torch.Tensor = ray_gt_dict['ray_ts'].to(device=self.device)  # (b, qo, ho, wo)
            gt_surface_normals_w: torch.Tensor = ray_gt_dict['surface_normals_w'].to(
                device=self.device)  # (b, qo, ho, wo, 3)
            if self.pcml_info.get('normalize_gt_normal', False):
                orientation = torch.ones((1, 3)).to(device=self.device)
                normal_sgn = (torch.sum(gt_surface_normals_w * orientation, dim=-1, keepdim=True) >= 0).float() * 2 - 1
                gt_surface_normals_w = gt_surface_normals_w * normal_sgn
            
            gt_hit_map: torch.Tensor = ray_gt_dict['hit_map'].to(device=self.device)  # (b, qo, ho, wo)
            valid_mask = gt_hit_map > 0.5
            invalid_mask = gt_hit_map <= 0.5
            input_point_cloud = batch['pcd'].to(device=self.device)
            valid_dict = input_point_cloud.get_true_valid_data()
            assert 'normal_w' in valid_dict.keys(), 'normal not in valid_dict'
            if self.pcml_info.get('normalize_point_normal', False):
                center = torch.ones((1, 3)).to(device=self.device) * 512
                outbound_orientation = [
                    xyz_w - center for xyz_w in valid_dict['xyz_w']
                ]
                outbound_normal_sgn = [
                    (torch.sum(valid_dict['normal_w'][i] * outbound_orientation[i], dim=-1, keepdim=True) > 0).float() * 2 - 1
                    for i in range(len(valid_dict['normal_w']))
                ]
                normalized_normal_w = [
                    valid_dict['normal_w'][i] * outbound_normal_sgn[i] for i in range(len(valid_dict['normal_w']))
                ]
                normal_w = normalized_normal_w
            else:
                normal_w = valid_dict['normal_w']
            
            rescale_input_pts_offset = self.dataset_info.get('rescale_input_pts_offset', 512)
            rescale_input_pts_scale = self.dataset_info.get('rescale_input_pts_scale', 256)
            
            if self.dataset_info['voxelized']:
                if self.dataset_info['random_rescale']:
                    input_scale_factors = [
                        np.random.choice(
                            [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375, 1.0], 1)[0] for i in range(self.batch_size)
                    ]
                else:
                    input_scale_factors = [1.0 for i in range(self.batch_size)]
                assert len(valid_dict['xyz_w']) == self.batch_size
                points_collate = [
                    torch.round(valid_dict['xyz_w'][i] * input_scale_factors[i]) /  input_scale_factors[i] for i in range(self.batch_size)
                ]
                color_feats = [
                    rgb.float().to(device=self.device) for rgb in valid_dict['rgb']
                ]
                normal_feats = [
                    normal_w[i].float().to(device=self.device) for i in range(self.batch_size)
                ]
            else:
                assert len(valid_dict['xyz_w']) == self.batch_size

                # First general full-quality (scaled) point cloud
                if self.dataset_info.get('rescale_input_pts', False):
                    points_collate = [
                        valid_dict['xyz_w'][i] * rescale_input_pts_scale + rescale_input_pts_offset for i in range(self.batch_size)
                    ]
                else:
                    points_collate = [
                        valid_dict['xyz_w'][i] for i in range(self.batch_size)
                    ]
                assert torch.min(points_collate[0]) >= 0
                
                # Adapt to different sparsity
                if self.dataset_info['random_rescale']:
                    if self.dataset_info.get('random_voxelization', 0.) <= np.random.rand():
                        # no voxelization
                        input_scale_factors = [
                            np.random.choice(
                                [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0], 1)[0] for i in range(self.batch_size)
                        ]

                        num_select_points_in_batch = [
                            int(points_collate[i].shape[0] * input_scale_factors[i]) for i in range(self.batch_size)
                        ]
                        subsample_indices = [
                            torch.randperm(points_collate[i].shape[0])[:num_select_points_in_batch[i]]
                            for i in range(self.batch_size)
                        ]

                        points_collate = [
                            points_collate[i][subsample_indices[i]] for i in range(self.batch_size)
                        ]
                        color_feats = [
                            valid_dict['rgb'][i].float().to(device=self.device)[subsample_indices[i]]
                            for i in range(len(valid_dict['rgb']))
                        ]
                        normal_feats = [
                            valid_dict['normal_w'][i].float().to(device=self.device)[subsample_indices[i]]
                            for i in range(len(valid_dict['normal_w']))
                        ]
                    else:
                        # voxelization
                        input_scale_factors = [
                            np.random.choice(
                                [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.9375, 1.0], 1)[0] for i in range(self.batch_size)
                        ]
                        points_collate = [
                            torch.round(points_collate[i] * input_scale_factors[i]) /  input_scale_factors[i] for i in range(self.batch_size)
                        ]
                        color_feats = [
                            rgb.float().to(device=self.device) for rgb in valid_dict['rgb']
                        ]
                        normal_feats = [
                            normal_w[i].float().to(device=self.device) for i in range(self.batch_size)
                        ]
                else:
                    # points_collate is ready
                    color_feats = [
                        rgb.float().to(device=self.device) for rgb in valid_dict['rgb']
                    ]
                    normal_feats = [
                        normal_w[i].float().to(device=self.device) for i in range(self.batch_size)
                    ]

            
            geom_feats = [
                xyz.float().to(device=self.device) for xyz in points_collate
            ]
            geom_quantize_offsets = [
                xyz - torch.round(xyz) for xyz in geom_feats
            ]
            # VOXEL_GRID_FEAT_NORMALIZER = 512.
            # geom_concat_feats = [
            #     torch.cat([xyz / VOXEL_GRID_FEAT_NORMALIZER, geom_quantize_offsets[i]], dim=-1) for i, xyz in enumerate(geom_feats)
            # ]

            geom_color_feats = [
                torch.cat([geom_quantize_offsets[i], color_feats[i]], dim=-1) for i in range(self.batch_size)
            ]
            gt_normal_feats = normal_feats
            geom_color_normal_feats = [
                torch.cat([geom_color_feats[i], gt_normal_feats[i]], dim=-1) for i in range(self.batch_size)
            ]


            geom_color_coords, geom_color_normal_feats = ME.utils.sparse_collate(
                points_collate, geom_color_normal_feats)
            geom_color_normal_sparse = ME.SparseTensor(
                features=geom_color_normal_feats, coordinates=geom_color_coords, device=self.device, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
            geom_color_reduced = geom_color_normal_sparse.F[..., :-3]
            normal_reduced = geom_color_normal_sparse.F[..., -3:]
            color_sparse = ME.SparseTensor(
                features=geom_color_reduced, device=self.device,
                coordinate_map_key=geom_color_normal_sparse.coordinate_map_key,
                coordinate_manager=geom_color_normal_sparse.coordinate_manager)
            normal_sparse = ME.SparseTensor(
                features=normal_reduced, device=self.device,
                coordinate_map_key=geom_color_normal_sparse.coordinate_map_key,
                coordinate_manager=geom_color_normal_sparse.coordinate_manager)
            _, gt_normal_reduced = normal_sparse.decomposed_coordinates_and_features
            
            decoded_primitives, decoded_sh, decoded_r, decoded_s, decoded_o, bpp, center_points, decoded_offsets, _, bpp_clr, bpp_hyper, decoded_n = self.pcml_model(color_sparse)
            
            ray_space_points_sparse_recon = [
                pcgc_rescale(
                    decoded_primitives[i].float(), offset=rescale_input_pts_offset, factor=self.pcml_info['scale_factor'], device=self.device)
                    for i in range(self.batch_size)
                    ]
            batchsize = len(ray_space_points_sparse_recon)

            # Get camera information
            gt_camera= batch['cam'].to(device=self.device)
            batchsize = gt_ray_rgb.shape[0]
            num_q = gt_ray_rgb.shape[1]
            camera_chunks = gt_camera.chunk(batchsize, dim=0)

            # Background
            if self.randomize_background:
                bg = torch.rand(3).to(device=self.device)
            else:
                bg = torch.ones(3).to(device=self.device)
            
            # Resolution
            if self.optim_info['random_multiscale']:
                scale = 2 ** np.random.randint(0, 3)
                # reshape and rescale
                _shape = gt_ray_rgb.shape
                gt_ray_rgb = gt_ray_rgb.permute(0, 1, 4, 2, 3)
                gt_ray_rgb = gt_ray_rgb.reshape(_shape[0] * _shape[1], _shape[4], _shape[2], _shape[3])
                gt_ray_rgb = F.interpolate(gt_ray_rgb, scale_factor=1/scale, mode='bicubic', align_corners=False)
                gt_ray_rgb = gt_ray_rgb.reshape(_shape[0], _shape[1], _shape[4], _shape[2] // scale, _shape[3] // scale)
                gt_ray_rgb = gt_ray_rgb.permute(0, 1, 3, 4, 2)

                gt_hit_map = gt_hit_map.reshape(_shape[0] * _shape[1], 1,  _shape[2], _shape[3])
                gt_hit_map = F.interpolate(gt_hit_map.float(), scale_factor=1/scale, mode='bilinear', align_corners=False).squeeze(1).long()
                gt_hit_map = gt_hit_map.reshape(_shape[0], _shape[1], _shape[2] // scale, _shape[3] // scale)
            render_width = gt_ray_rgb.shape[3]
            render_height = gt_ray_rgb.shape[2]
        
            gt_ray_rgb = gt_ray_rgb * gt_hit_map.unsqueeze(-1) + (1 - gt_hit_map.unsqueeze(-1)) * bg.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            effective_gt_hit_map = gt_hit_map
            gt_hit_map = torch.ones_like(gt_hit_map)

            # render color and optionally normal, xyz_w
            rendered_recons = self._rasterize(
                batchsize,
                ray_space_points_sparse_recon,
                dict(
                    decoded_o=decoded_o,
                    decoded_s=decoded_s,
                    decoded_r=decoded_r,
                    decoded_sh=decoded_sh,
                    colors_precomp=None,
                ),
                camera_chunks,
                num_q,
                bg,
                gt_ray_rgb.shape[2],
                gt_ray_rgb.shape[3],
            )


            rendered_normal = self._rasterize(
                batchsize,
                ray_space_points_sparse_recon,
                dict(
                    decoded_o=decoded_o,
                    decoded_s=decoded_s,
                    decoded_r=decoded_r,
                    decoded_sh=None,
                    colors_precomp=decoded_n,
                ),
                camera_chunks,
                num_q,
                torch.ones_like(bg),
                gt_ray_rgb.shape[2],
                gt_ray_rgb.shape[3],
                normalize_camera_normal=self.pcml_info.get('normalize_camera_normal', False),
            )

            rendered_xyz_w = self._rasterize(
                batchsize,
                ray_space_points_sparse_recon,
                dict(
                    decoded_o=decoded_o,
                    decoded_s=decoded_s,
                    decoded_r=decoded_r,
                    decoded_sh=None,
                    colors_precomp=ray_space_points_sparse_recon,
                ),
                camera_chunks,
                num_q,
                torch.zeros_like(bg),
                gt_ray_rgb.shape[2],
                gt_ray_rgb.shape[3],
            )


            intersection_xyz_w = rendered_xyz_w
            intersection_surface_normal_w = rendered_normal
            intersection_rgb = rendered_recons
            pointersect_record = structures.PointersectRecord(
                intersection_xyz_w=intersection_xyz_w,
                intersection_surface_normal_w=intersection_surface_normal_w,
                intersection_rgb=intersection_rgb,
                blending_weights=None,
                neighbor_point_idxs=None,
                neighbor_point_valid_len=None,
                ray_t=intersection_xyz_w[..., 0],
                ray_hit=gt_hit_map,
                ray_hit_logit=gt_hit_map,
                model_attn_weights=None,
            )

            out_dict = {
                'pointersect_record': pointersect_record
            }
            self.pointersect_record: structures.PointersectRecord = out_dict['pointersect_record']
                
            gt_rgbd = structures.RGBDImage(
                gt_ray_rgb, gt_ray_ts, gt_camera, gt_surface_normals_w, gt_hit_map)
            irgb = self.pointersect_record.intersection_rgb
            self.loss_total = 0.
            if True:
                assert self.pointersect_record.intersection_rgb is not None
              
                if self.optim_info['loss_rgb_type'] == 'l2':
                    self.loss_rgb = self.loss_mse(
                        gt_ray_rgb,  # (b, qo, ho, wo, 3)
                    self.pointersect_record.intersection_rgb,  # (b, qo, ho, wo, 3)
                    )  # (b, m, 3)
                elif self.optim_info['loss_rgb_type'] == 'l1':
                    self.loss_rgb = self.loss_l1(
                        gt_ray_rgb,  # (b, qo, ho, wo, 3)
                        self.pointersect_record.intersection_rgb,  # (b, qo, ho, wo, 3)
                    )  # (b, m, 3)
                else:
                    raise NotImplementedError
            self.loss_rgb  = self.loss_rgb.sum() / (effective_gt_hit_map.sum() + 1e-6)
            self.loss_total += self.loss_rgb

            # Calculate decoded_n loss
            if self.optim_info.get('loss_weight_point_normal', 0) > 0:
                point_normal_l1 = [
                    self.loss_l1(
                        decoded_n[i],
                        gt_normal_reduced[i],
                    ).mean() for i in range(self.batch_size)
                ]
                point_normal_l1 = torch.stack(point_normal_l1, dim=0)
                self.loss_point_normal_l1 = point_normal_l1.mean()
                self.loss_total += self.loss_point_normal_l1 * self.optim_info['loss_weight_point_normal']

            if self.optim_info.get('loss_weight_normal', 0):
                self.loss_normal = torch.cross(
                    self.pointersect_record.intersection_surface_normal_w,  # (b, qo, ho, wo, 3)
                    gt_surface_normals_w.clone().masked_fill_(invalid_mask.unsqueeze(-1), 1.),  # (b, qo, ho, wo, 3)
                    dim=-1,
                ).pow(2).sum() / (effective_gt_hit_map.sum() + 1e-6)  # (b, m)  sin(theta)^2

                self.loss_normal_l1 = self.loss_l1(
                    self.pointersect_record.intersection_surface_normal_w,
                    gt_surface_normals_w.clone().masked_fill_(invalid_mask.unsqueeze(-1), 1.)
                ).sum() / (effective_gt_hit_map.sum() + 1e-6)

                loss_normal_l2_side_1 = self.loss_mse(
                    self.pointersect_record.intersection_surface_normal_w,
                    gt_surface_normals_w.clone().masked_fill_(invalid_mask.unsqueeze(-1), 1.)
                ) / (effective_gt_hit_map.sum() + 1e-6)

                loss_normal_l2_side_2 = self.loss_mse(
                    self.pointersect_record.intersection_surface_normal_w * -1,
                    gt_surface_normals_w.clone().masked_fill_(invalid_mask.unsqueeze(-1), 1.)
                ) / (effective_gt_hit_map.sum() + 1e-6)

                self.loss_normal_l2 = torch.min(loss_normal_l2_side_1, loss_normal_l2_side_2).sum()

                self.loss_normal_reg = (((self.pointersect_record.intersection_surface_normal_w ** 2).sum(-1) - 1) ** 2).mean()
                
                self.loss_total += self.loss_normal * self.optim_info['loss_weight_normal'] + self.loss_normal_l1 * self.optim_info['loss_weight_normal_l1'] + self.loss_normal_l2 * self.optim_info['loss_weight_normal_l2'] + self.loss_normal_reg * self.optim_info['loss_weight_normal_reg']
            
            if self.optim_info['loss_weight_lpips'] > 0:
                b, q, h, w, c = gt_ray_rgb.shape
                organized_gt_ray_rgb = gt_ray_rgb.reshape((b * q, h, w, c)).permute(0, 3, 1, 2)
                organized_intersection_rgb = self.pointersect_record.intersection_rgb.reshape((b * q, h, w, c)).permute(0, 3, 1, 2)
                self.loss_lpips = self.lpips(
                    organized_gt_ray_rgb, organized_intersection_rgb
                    ).mean() * self.optim_info['loss_weight_lpips']
                self.loss_total += self.loss_lpips
            else:
                self.loss_lpips = 0.
            if self.optim_info['loss_weight_ssim'] > 0:
                b, q, h, w, c = gt_ray_rgb.shape
                organized_gt_ray_rgb = gt_ray_rgb.reshape((b * q, h, w, c)).permute(0, 3, 1, 2)
                organized_intersection_rgb = self.pointersect_record.intersection_rgb.reshape((b * q, h, w, c)).permute(0, 3, 1, 2)
                self.loss_ssim = 1 - ssim(
                    organized_gt_ray_rgb, organized_intersection_rgb
                    ).mean() * self.optim_info['loss_weight_ssim']
                self.loss_total += self.loss_ssim
            else:
                self.loss_ssim = 0.

            self.loss_bpp_clr = bpp_clr
            self.loss_bpp_hyper = bpp_hyper


            if update:
                self.optimizer.zero_grad()
                # with torch.autograd.set_detect_anomaly(True): 
                self.scaler.scale(self.loss_total).backward()
                self.scaler.unscale_(self.optimizer) 
                torch.nn.utils.clip_grad_norm_(self.update_paras, self.optim_info["clip"])
                self.scaler.step(self.optimizer)
                # self.scheduler.step()
                self.scaler.update()
            #save pcd and images
            if self.process_info['rank'] == 0:
                if bidx % self.process_info['save_pic_every_batch'] == 0 or \
                    (bidx % self.process_info['visualize_every_num_test_batch'] == 0 and not update):
                    pic_dir = os.path.join(self.process_info["output_dir"], f'training_result',self.process_info["exp_tag"])\
                        if not self.process_info["exp_tag_first"] else os.path.join(self.process_info["output_dir"],self.process_info["exp_tag"], f'training_result')
                    pic_dir = os.path.join(pic_dir,str(epoch)+"_"+str(bidx)) if update \
                        else os.path.join(pic_dir,str(epoch)+"_test")
                    res_dir = os.path.join(pic_dir, f'result')
                    #pcgc_pcd = structures.PointCloud(xyz_w=upsampled_p,normal_w=upsampled_np)
                    
                    rgbd_images_pointersect = self.pointersect_record.get_rgbd_image(camera=gt_camera)
                    rgbd_images_pointersect.save(
                        output_dir=res_dir,
                        overwrite=True,
                        save_png=True,
                        save_gif=False,
                        save_video=False,
                        save_rgb=False,
                        hit_only=False,
                        save_pt=False
                    )
                    
                    gt_dir = os.path.join(pic_dir, f'gt')
                    # input_point_cloud.save(
                    #     gt_dir,
                    #     True,
                    #     True,
                    #     False
                    # )
                    gt_rgbd.save(
                        gt_dir,
                        overwrite=True,
                        save_gif=False,
                        save_video=False,
                        save_rgb=False,
                        hit_only=False,
                        save_pt=False
                    )
                

            res = dict(
                loss = self.loss_total,
                bpp = bpp,
                loss_bpp_clr = self.loss_bpp_clr,
                loss_bpp_hyper = self.loss_bpp_hyper,
                loss_normal = self.loss_normal,
                loss_normal_l1 = self.loss_normal_l1,
                loss_normal_l2 = self.loss_normal_l2,
                loss_normal_reg = self.loss_normal_reg
            )
            
            pts_res = dict(                        
                loss_weight_t = self.loss_t,
                loss_normal = self.loss_normal,
                loss_weight_hit = self.loss_hit,
                loss_dc = self.loss_dc,
                loss_lpips=self.loss_lpips,
                loss_cd=self.loss_cd,
                )
            pts_res.update(dict(
                loss_rgb = self.loss_rgb,
                loss_weight_point_normal_l1 = self.loss_point_normal_l1,
                loss_ssim = self.loss_ssim,
            ))
            res.update(pts_res)
            torch.cuda.empty_cache()
            return res
    
    def get_current_lrs(self):
        """
        Get a dictionary containing the current learning rate of each optimizer
        """
        lr_dict = dict()
        lr_dict['lr'] = self._optimizer.param_groups[0]['lr']
        return lr_dict

    def _register_output(
            self,
            var_name: T.Union[str, T.List[str]],
    ):
        """
        Register output so that it will be reset by `reset_outputs`

        Args:
            var_name:
                var name (str) or list of var names
        """
        self._register(var_name=var_name, target=self.outputs_name)

    def _register_loss(
            self,
            var_name: T.Union[str, T.List[str]],
    ):
        """
        Register scalar loss so that it will be gathered by :py:`_gather_loss`

        Args:
            var_name:
                loss name (str) or list of loss names
        """
        self._register(var_name=var_name, target=self.losses_name)

    def _register(
            self,
            var_name: T.Union[str, T.List[str]],
            target: T.Set[str],
    ):
        """
        Register the var_name to the target (set of var names)

        Args:
            var_name:
                name (str) or list of names of the variable to register
            target:
                the set to register the var namee
        """
        if isinstance(var_name, str):
            var_name = [var_name]

        for name in var_name:
            target.add(name)

    def _gather_losses_by_losses_name(self, return_float=True) -> T.Dict[str, float]:
        """
        Return the values in losses_name in a dictionary.
        """
        loss_dict = dict()

        if not hasattr(self, 'losses_name'):
            return loss_dict

        for name in sorted(self.losses_name):
            loss = getattr(self, name, None)
            if loss is None:
                continue
            else:
                loss_dict[name] = loss

        loss_dict = self._gather_losses(
            loss_dict=loss_dict,
            return_float=return_float,
        )
        return loss_dict

    def _gather_losses(
            self,
            loss_dict: T.Dict[str, torch.Tensor],
            return_float: bool = True,
    ) -> T.Dict[str, float]:
        """
        Return the values in losses_name in a dictionary.
        """
        if loss_dict is None:
            return None

        out_dict = dict()
        keys = sorted(list(loss_dict.keys()))
        for name in keys:  # sorted to support dist.reduce
            val = loss_dict[name]
            if val is None:
                continue

            if not isinstance(val, torch.Tensor):
                out_dict[name] = val
            else:
                out_dict[name] = val.detach()

        if return_float:
            for key, val in out_dict.items():
                if isinstance(val, torch.Tensor):
                    out_dict[key] = val.detach().cpu().item()
                elif isinstance(val, np.ndarray):
                    out_dict[key] = val.item()
                elif isinstance(val, int):
                    out_dict[key] = float(val)

        return out_dict

    def reset_outputs(self):
        """
        Set the outputs of the model to None.
        """
        for name in self.outputs_name:
            setattr(self, name, None)

    def reset_losses(self):
        """
        Set the losses to None.
        """
        for name in self.losses_name:
            setattr(self, name, None)

    def train_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ):

        self.reset_losses()
        self.reset_outputs()

        all_loss_dict = self._step(
            epoch=epoch,
            bidx=bidx,
            batch=batch,
            update=True,
        )
        return all_loss_dict

    def validation_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ):
        self.reset_losses()
        self.reset_outputs()

        all_loss_dict = self._step(
            epoch=epoch,
            bidx=bidx,
            batch=batch,
            update=False,
        )
        return all_loss_dict

    def test_step(
            self,
            epoch: int,
            bidx: int,
            batch: T.Any,
    ):
        self.reset_losses()
        self.reset_outputs()

        all_loss_dict = self._step(
            epoch=epoch,
            bidx=bidx,
            batch=batch,
            update=False,
        )
        return all_loss_dict

    def visualize_train_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f'batch uses: {batch_time:.4f} secs, '
            f'(step uses: {step_time:.4f} secs, '
            f'{step_time / batch_time * 100.:.2f}%)'
        )

    def visualize_validation_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f'batch uses: {batch_time:.4f} secs, '
            f'(step uses: {step_time:.4f} secs, '
            f'{step_time / batch_time * 100.:.2f}%)'
        )


    def visualize_test_step(
            self,
            epoch: int,
            total_batch_count: int,
            bidx: int,
            batch: T.Any,
            out_dict: T.Dict[str, T.Any],
            logger: Logger,
            batch_time: float,
            step_time: float,
            epoch_time: float,
    ):
        self.logger.info(
            f'batch uses: {batch_time:.4f} secs, '
            f'(step uses: {step_time:.4f} secs, '
            f'{step_time / batch_time * 100.:.2f}%)'
        )
    def epoch_setup(
            self,
            epoch: int,
            dataloader: T.Sequence[T.Any],
            val_dataloader: T.Sequence[T.Any],
            test_dataloader: T.Sequence[T.Any],
    ):
        """Set up at the beginning of an epoch, before dataloder iterator
         is constructed.  It can be used to setup the batch sampler, etc."""

        # used when distribution learning
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        if self.valid_sampler is not None:
            self.valid_sampler.set_epoch(epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(epoch)




if __name__ == "__main__":
    matplotlib.use('agg')
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument(
        '--trainer_filename',
        type=str,
        default=None,
        help='previous pth file')
    parser.add_argument(
        '--config_filename',
        type=str,
        default=None,
        help='config yaml/json file')
    parser.add_argument(
        '--num_threads',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--pcd_subsample',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--k',
        type=int,
        default=40,
    )
    parser.add_argument(
        '--visualize_every_num_train_batch',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--visualize_every_num_valid_batch',
        type=int,
        default=30,
    )
    parser.add_argument(
        '--visualize_every_num_test_batch',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--skip_existed_cleaned_mesh',
        action='store_true',
    )

    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--n_gpus', type=int, default=1)
    options = parser.parse_args()

    # torch.autograd.set_detect_anomaly(True)

    with TrainPointersectProcess(
            trainer_filename=options.trainer_filename,
            config_filename=options.config_filename,
            rank=options.rank,
            n_gpus=options.n_gpus,
            save_code=False,  # will be overwritten by config_filename if given
            pcd_subsample=options.pcd_subsample,  # will be overwritten by config_filename if given
            num_threads=options.num_threads,  # will be overwritten by config_filename if given
            k=options.k,  # will be overwritten by config_filename if given
            visualize_every_num_train_batch=options.visualize_every_num_train_batch,
            visualize_every_num_valid_batch=options.visualize_every_num_valid_batch,
            visualize_every_num_test_batch=options.visualize_every_num_test_batch,
            skip_existed_cleaned_mesh=options.skip_existed_cleaned_mesh,  # just for speed, default false
    ) as trainer:
        trainer.run()
