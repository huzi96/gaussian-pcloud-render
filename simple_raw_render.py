import time

import typing as T
import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from structures import *
import math,yaml

import open3d as o3d
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from models.sh_utils import *
from models.model_v2 import PCEncoder


def generate_cam(camera_info, save_temp_state_dict=True, return_traj=False):
    output_camera_trajectory_params_defaults = {
        'min_r': 3,
        'max_r': 4,
        'max_angle': 30.,
        'num_circle': 4,
        'r_freq': 1,
        'max_translate_ratio':2.,
        'local_max_angle':3.,
        'rand_r':0.,
        'H_c2w': torch.eye(4).unsqueeze(0).repeat(2, 1, 1).unsqueeze(0)
    }

    output_cam_trajectory = CameraTrajectory(
        mode=camera_info['mode'],
        n_imgs=camera_info['n_imgs'],
        total=1,
        rng_seed=0,
        params=camera_info if camera_info['mode'] != 'udlrfb' else output_camera_trajectory_params_defaults,

    )
    output_cameras = output_cam_trajectory.get_camera(
        fov=camera_info['fov'],
        width_px=camera_info['width_px'],
        height_px=camera_info['height_px'],
        device=torch.device('cpu'),
    )
    if save_temp_state_dict:
        torch.save(output_cameras.state_dict(), 'validate/temp_state_dict.pt')
    if return_traj:
        return output_cameras, output_cam_trajectory
    else:
        return output_cameras

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

def pcgc_rescale(input_xyz, offset=512, factor=256, device=None):
    pnum = input_xyz.shape[0]
    res = (input_xyz - offset) / factor
    device = input_xyz.device
    return res.to(device)

def get_rasterize_param_from_camera(
        camera: Camera,device:torch.device, fovX_deg=40., fovY_deg=40.,\
        sh_degree=0, bg=None, super_sample_rate=2):
    # view_mat = getWorld2View2(np.eye(3), np.ones(3) * 5., scale=1.0)
        world_view_transform = camera.get_H_w2c().to(device=device).transpose(-2, -1)
        projection_matrix = getProjectionMatrix(
            znear=0.01, zfar=100,
            fovX=np.pi * fovX_deg / 180, fovY=np.pi * fovY_deg / 180,
            )
        projection_matrix = projection_matrix.transpose(-2, -1).cuda()
        mat_shape = world_view_transform.shape
        world_view_transform = world_view_transform.reshape(-1, 4, 4)
        projection_matrix = projection_matrix.unsqueeze(0).expand(world_view_transform.shape[0], -1, -1).reshape(-1, 4, 4)
        full_proj_transform = (
            world_view_transform.bmm(projection_matrix))
        if bg is None:
            bg = torch.zeros((3,)).to(device=device)
        else:
            bg = bg.to(device=device)
        rasterize_setting = GaussianRasterizationSettings(
            image_height = camera.height_px * super_sample_rate,
            image_width = camera.width_px * super_sample_rate,
            tanfovx = math.tan(fovX_deg / 180. * math.pi),
            tanfovy = math.tan(fovY_deg / 180. * math.pi),
            bg = bg,
            scale_modifier = 1.0,
            viewmatrix =world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree = sh_degree,
            campos = torch.matmul(camera.H_c2w, torch.tensor([0, 0, 0, 1]).to(device=device).float())[..., 0:3],
            prefiltered = False,
            debug = False
        )
        return rasterize_setting

def load_pcml(ckpt:str, ModelClass):
    pth = ckpt.split('/')
    root_pth = pth[:-2]
    opt = ['option','options.yaml']
    opt_pth = '/'.join(root_pth+opt)
    print(opt_pth)

    opt_file = open(opt_pth, 'r')
    data = yaml.load(opt_file, Loader=yaml.FullLoader)
    info = data['pcml_info']
    opt_file.close()

    model = ModelClass(info)
    pcml_ckpt = torch.load(ckpt)
    model.load_state_dict(pcml_ckpt)
    print('Loaded weights.')
    return model,info

def save_pic(img:torch.Tensor, pth:str, type='rgb', hit_map=None, suffix=''):
    os.makedirs(pth, exist_ok=True)
    b, q, h, w, _3 = img.shape
    for ib in range(b):
        for iq in range(q):
            filename = os.path.join(pth, f'{type}_{iq}{suffix}.png')
            if type == 'rgb':
                imageio.imwrite(
                    filename,
                    (img[ib, iq] * 255.).detach().cpu().clamp(min=0, max=255).numpy().astype(np.uint8)
                )
            elif type == 'normal_w':
                if hit_map is None:
                    imageio.imwrite(
                        filename,
                        (
                            ((img[ib, iq] + 1) / 2.)  * 255.).detach().cpu().clamp(min=0, max=255).numpy().astype(np.uint8)
                    )
                else:
                    imageio.imwrite(
                        filename,
                        (
                            (((img[ib, iq] + 1) / 2.) * hit_map[ib, iq] + (1 - hit_map[ib, iq]))  * 255.).detach().cpu().clamp(min=0, max=255).numpy().astype(np.uint8)
                    )
            elif type == 'xyz_w':
                imageio.imwrite(
                    filename,
                    ((img[ib, iq] + 1) / 2. * 255.).detach().cpu().clamp(min=0, max=255).numpy().astype(np.uint8)
                )
            elif type == 'shaded':
                imageio.imwrite(
                    filename,
                    (img[ib, iq] * 255.).detach().cpu().clamp(min=0, max=255).numpy().astype(np.uint8)
                )

class PCML_Render:
    #render raw point cloud with rgb
    def __init__(self, ckpt:str, voxelized=True, scale_factor=None, offset=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.info = load_pcml(ckpt, PCEncoder)
        self.model = self.model.to(self.device)
        self.voxelized = voxelized
        if scale_factor is None:
            self.scale_factor = self.info['scale_factor']
        else:
            self.scale_factor = scale_factor
        self.offset = offset
        
    
    def _self_build_rotation(self, r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R = torch.stack([
            1 - 2 * (y*y + z*z),
            2 * (x*y - r*z),
            2 * (x*z + r*y),
            2 * (x*y + r*z),
            1 - 2 * (x*x + z*z),
            2 * (y*z - r*x),
            2 * (x*z - r*y),
            2 * (y*z + r*x),
            1 - 2 * (x*x + y*y),
        ], dim=1)
        R = R.reshape(-1, 3, 3)
        return R

    
    def _est_normal_from_ellipsoid(self, decoded_s, decoded_r):
        normal_template = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float32).to(device=self.device)
        normal_template = normal_template
        decoded_n = []
        for i in range(len(decoded_s)):
            s = decoded_s[i] # (N, 3)
            r = decoded_r[i] # (N, 4)
            print('R transposed.')
            R = self._self_build_rotation(r).permute(0, 2, 1)
            n_indices = torch.min(s, dim=1)[1]
            normals = []
            for j in range(len(s)):
                normal_j = R[j, n_indices[j]]
                normals.append(normal_j)
            normals = torch.stack(normals, dim=0)
            decoded_n.append(normals)
        return decoded_n

    
    def _rasterize(self, batchsize,
                   ray_space_points_sparse_recon, primitive_dict,
                   camera_chunks, num_q, bg, h, w,
                   super_sample_rate, fov,
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
            if self.info.get('enable_opacity', True):
                opacity = decoded_o[i]
            else:
                print('Warning: opacity is disabled.')
                opacity = torch.ones_like(decoded_o[i])
            radius = np.sqrt(3) / self.scale_factor * 6
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
                raster_settings = get_rasterize_param_from_camera(
                    camera_chunk_j[j], bg=bg, sh_degree=self.info['sh_deg'], fovX_deg=fov, fovY_deg=fov, device=self.device, super_sample_rate=super_sample_rate)
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
        rendered_recons = rendered_recons.reshape(batchsize * num_q, 3, h * super_sample_rate, w * super_sample_rate)
        if super_sample_rate > 1:
            rendered_recons = F.interpolate(
                rendered_recons, size=(h, w),
                mode='bilinear', align_corners=False)
        rendered_recons = rendered_recons.reshape(
            batchsize, num_q, 3, h, w)
        rendered_recons = rendered_recons.permute(0, 1, 3, 4, 2)
        return rendered_recons
    
    def render(self, pcd:PointCloud, scale:int, cam:Camera, fov:int,
               enable_opacity:True, super_sample_rate=2, input_offset=None,
               point_light=None, consistent_normal=False,
               est_normal_from_ellipsoid=False, background_color=0.):
        in_dim = int(self.info['clr_encoder_channels'].split(' ')[0])
        if input_offset is None:
            in_offset = torch.zeros(1, 3).to(device=pcd.xyz_w[0].device)
        else:
            in_offset = torch.tensor(input_offset, dtype=torch.float32).to(device=pcd.xyz_w[0].device)
        if in_dim == 3:
            points_collate = [
                pcd_points + in_offset for pcd_points in pcd.xyz_w
                ]
            color_feats = [
                rgb.float().to(device=self.device) for rgb in pcd.rgb
                ]
            color_coords, color_feats = ME.utils.sparse_collate(
                    points_collate, color_feats)
            color_sparse = ME.SparseTensor(
                    features=color_feats, coordinates=color_coords, device=self.device)
        elif in_dim == 9:
            if self.voxelized:
                points_collate = [
                    pcd_points + in_offset for pcd_points in pcd.xyz_w
                ]
            else:
                points_collate = [
                    pcd_points * self.scale_factor + self.offset + in_offset for pcd_points in pcd.xyz_w
                ]
            geom_feats = [
                xyz.float().to(device=self.device) for xyz in points_collate
            ]
            geom_quantize_offsets = [
                xyz - torch.round(xyz) for xyz in geom_feats
            ]
            geom_concat_feats = [
                torch.cat([(xyz - self.offset) / self.scale_factor, geom_quantize_offsets[i]], dim=-1) for i, xyz in enumerate(geom_feats)
            ]
            color_feats = [
                rgb.float().to(device=self.device) for rgb in pcd.rgb
            ]
            geom_color_feats = [
                torch.cat([geom_concat_feats[i], color_feats[i]], dim=-1) for i in range(len(geom_concat_feats))
            ]

            geom_color_coords, geom_color_feats = ME.utils.sparse_collate(
                points_collate, geom_color_feats)
            color_sparse = ME.SparseTensor(
                features=geom_color_feats, coordinates=geom_color_coords, device=self.device, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        elif in_dim == 6:
            if self.voxelized:
                points_collate = [
                    pcd_points + in_offset for pcd_points in pcd.xyz_w
                ]
            else:
                points_collate = [
                    pcd_points * self.scale_factor + self.offset + in_offset for pcd_points in pcd.xyz_w
                ]

            geom_feats = [
                xyz.float().to(device=self.device) for xyz in points_collate
            ]
            geom_quantize_offsets = [
                xyz - torch.round(xyz) for xyz in geom_feats
            ]
            color_feats = [
                rgb.float().to(device=self.device) for rgb in pcd.rgb
            ]
            geom_color_feats = [
                torch.cat([geom_quantize_offsets[i], color_feats[i]], dim=-1) for i in range(len(color_feats))
            ]

            geom_color_coords, geom_color_feats = ME.utils.sparse_collate(
                points_collate, geom_color_feats)
            
            color_sparse = ME.SparseTensor(
                features=geom_color_feats, coordinates=geom_color_coords, device=self.device, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)

        else:
            raise NotImplementedError

        # Model Warmup
        _ = self.model(color_sparse)
        torch.cuda.synchronize()
        start_time = time.time()
        decoded_primitives, decoded_sh, decoded_r, decoded_s, decoded_o, bpp, \
                center_points, decoded_offsets, _, bpp_clr, bpp_hyper, decoded_n = self.model(color_sparse)
        torch.cuda.synchronize()
        model_time = time.time() - start_time
        
        # decoded_primitives = center_points
        ########################################################################
        decoded_primitives = decoded_primitives[0:1]
        decoded_sh = decoded_sh[0:1]
        decoded_r = decoded_r[0:1]
        decoded_s = decoded_s[0:1]
        decoded_o = decoded_o[0:1]
        decoded_n = decoded_n[0:1]
        ########################################################################

        ray_space_points_sparse_recon = [
                pcgc_rescale(
                    decoded_primitives[i].float(), self.offset, self.scale_factor, device=self.device)
                    for i in range(len(decoded_primitives))
                    ]
        if not enable_opacity:
            print('Warning: opacity is disabled.')
            decoded_o = [torch.ones_like(otensor) for otensor in decoded_o]
        # Render
        batchsize = len(ray_space_points_sparse_recon)

        gt_camera= cam.to(self.device)
        num_q = gt_camera.H_c2w.shape[1]
        camera_chunks = gt_camera.chunk(batchsize, dim=0)

        rendered_recons = []
        radii_recons = []
        bg = torch.zeros(3).to(device=self.device) + background_color

        # Render world xyz
        primitive_dict = {
            'decoded_o': decoded_o,
            'decoded_s': decoded_s,
            'decoded_r': decoded_r,
            'decoded_sh': None,
            'colors_precomp': ray_space_points_sparse_recon,
        }

        rendered_xyz = self._rasterize(
            batchsize,
            ray_space_points_sparse_recon,
            primitive_dict,
            camera_chunks,
            num_q,
            bg,
            cam.height_px,
            cam.width_px,
            super_sample_rate,
            fov,
        )

        # Render RGB
        start_time = time.time()
        torch.cuda.synchronize()
        primitive_dict = {
            'decoded_o': decoded_o,
            'decoded_s': decoded_s,
            'decoded_r': decoded_r,
            'decoded_sh': decoded_sh,
            'colors_precomp': None,
        }
        rendered_recons = self._rasterize(
            batchsize,
            ray_space_points_sparse_recon,
            primitive_dict,
            camera_chunks,
            num_q,
            bg,
            cam.height_px,
            cam.width_px,
            super_sample_rate,
            fov,
        )
        torch.cuda.synchronize()
        rgb_time = time.time() - start_time
        print('model time: %.3f sec, rgb time: %.3f sec' % (model_time, rgb_time), flush=True)


        # Render hitmap
        primitive_dict = {
            'decoded_o': decoded_o,
            'decoded_s': decoded_s,
            'decoded_r': decoded_r,
            'decoded_sh': None,
            'colors_precomp': [torch.ones_like(ray_space_points_sparse_recon[i]) for i in range(len(ray_space_points_sparse_recon))],
        }
        rendered_hitmap = self._rasterize(
            batchsize,
            ray_space_points_sparse_recon,
            primitive_dict,
            camera_chunks,
            num_q,
            bg,
            cam.height_px,
            cam.width_px,
            super_sample_rate,
            fov
        )

        # Render normal (optional)
        if decoded_n is not None:
            decoded_consistent_n = []
            assert not consistent_normal
            if consistent_normal:
                for i in range(len(decoded_n)):
                    pts_np = ray_space_points_sparse_recon[i].cpu().numpy()
                    pts_np = pts_np.reshape(-1, 3)
                    norm_np = decoded_n[i].cpu().numpy()
                    norm_np = norm_np.reshape(-1, 3)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_np)
                    pcd.normals = o3d.utility.Vector3dVector(norm_np)
                    pcd.orient_normals_consistent_tangent_plane(100)
                    norm_np = np.asarray(pcd.normals)
                    norm_np = norm_np.reshape(1, -1, 3)
                    decoded_consistent_n.append(torch.tensor(norm_np, dtype=torch.float32).to(device=self.device))
            elif est_normal_from_ellipsoid:
                decoded_consistent_n = self._est_normal_from_ellipsoid(decoded_s, decoded_r)
            else:
                decoded_consistent_n = decoded_n
                
            primitive_dict = {
                'decoded_o': decoded_o,
                'decoded_s': decoded_s,
                'decoded_r': decoded_r,
                'decoded_sh': None,
                'colors_precomp': decoded_consistent_n,
            }
            # process decoded_n
            rendered_normals = self._rasterize(
                batchsize,
                ray_space_points_sparse_recon,
                primitive_dict,
                camera_chunks,
                num_q,
                bg,
                cam.height_px,
                cam.width_px,
                super_sample_rate,
                fov,
                normalize_camera_normal=True,
            )
        else:
            rendered_normals = None
        ret_dict = {
            'rgb': rendered_recons,
            'normal': rendered_normals,
            'xyz_w': rendered_xyz,
            'hitmap': rendered_hitmap,
        }
        
        if point_light is not None:
            lighted_maps = [ret_dict['rgb'] * point_light['light_coeff'][0]]
            for i in range(len(point_light['xyz_w'])):
                light_dir = ret_dict['xyz_w'] - point_light['xyz_w'][i].to(self.device)
                light_dir = light_dir / torch.norm(light_dir, dim=-1, keepdim=True)
                cos_theta = torch.sum(light_dir * ret_dict['normal'], dim=-1, keepdim=True)
                cos_theta = torch.clamp(cos_theta, min=0)
                # cos_theta = torch.abs(cos_theta)
                lighted_i = point_light['color'][i].to(self.device) * cos_theta * ret_dict['hitmap'] * ret_dict['rgb'] * point_light['light_coeff'][i+1]
                lighted_maps.append(lighted_i)
            ret_dict['shaded'] = torch.sum(torch.stack(lighted_maps, dim=0), dim=0)
        
        
        return ret_dict 

class Simple_Render:
    # Est global variance and render
    def __init__(self, voxelized=True, scale_factor=None, offset=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.voxelized = voxelized
        if scale_factor is None:
            self.scale_factor = 1.
        else:
            self.scale_factor = scale_factor
        self.offset = offset
        self.default_quaternion = torch.tensor([1, 0, 0, 0], dtype=torch.float32).to(device=self.device)
    
    def _self_build_rotation(self, r):
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R = torch.stack([
            1 - 2 * (y*y + z*z),
            2 * (x*y - r*z),
            2 * (x*z + r*y),
            2 * (x*y + r*z),
            1 - 2 * (x*x + z*z),
            2 * (y*z - r*x),
            2 * (x*z - r*y),
            2 * (y*z + r*x),
            1 - 2 * (x*x + y*y),
        ], dim=1)
        R = R.reshape(-1, 3, 3)
        return R

    def _est_normal_from_ellipsoid(self, decoded_s, decoded_r):
        normal_template = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float32).to(device=self.device)
        normal_template = normal_template
        decoded_n = []
        for i in range(len(decoded_s)):
            s = decoded_s[i] # (N, 3)
            r = decoded_r[i] # (N, 4)
            R = self._self_build_rotation(r)
            n_indices = torch.min(s, dim=1)[1]
            raw_normals = normal_template[n_indices]
            normals = torch.einsum('bij,bi->bj', R, raw_normals)
            decoded_n.append(normals)
        return decoded_n

    def _rasterize(self, batchsize,
                   ray_space_points_sparse_recon, primitive_dict,
                   camera_chunks, num_q, bg, h, w,
                   super_sample_rate, fov,
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
            opacity = torch.ones_like(decoded_o[i])
            # radius = np.sqrt(3) / self.scale_factor * 6
            scales = decoded_s[i]
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
                raster_settings = get_rasterize_param_from_camera(
                    camera_chunk_j[j], bg=bg, sh_degree=1, fovX_deg=fov, fovY_deg=fov, device=self.device, super_sample_rate=super_sample_rate)
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
        rendered_recons = rendered_recons.reshape(batchsize * num_q, 3, h * super_sample_rate, w * super_sample_rate)
        if super_sample_rate > 1:
            rendered_recons = F.interpolate(
                rendered_recons, size=(h, w),
                mode='bilinear', align_corners=False)
        rendered_recons = rendered_recons.reshape(
            batchsize, num_q, 3, h, w)
        rendered_recons = rendered_recons.permute(0, 1, 3, 4, 2)
        return rendered_recons
    
    def render(self, pcd:PointCloud, scale:int, cam:Camera, fov:int,
               enable_opacity:True, super_sample_rate=2, input_offset=None,
               point_light=None, consistent_normal=False,
               est_normal_from_ellipsoid=False, background_color=0.,
               sigma=1.):

        if input_offset is None:
            in_offset = torch.zeros(1, 3).to(device=pcd.xyz_w[0].device)
        else:
            in_offset = torch.tensor(input_offset, dtype=torch.float32).to(device=pcd.xyz_w[0].device)

        points_collate = [
            pcd_points + in_offset for pcd_points in pcd.xyz_w
            ]
        points_collate = [x.to(self.device) for x in points_collate]
        color_feats = [
            rgb.float().to(device=self.device) for rgb in pcd.rgb
            ]
        # normal_w_feats = [
        #     normal_w.float().to(device=self.device) for normal_w in pcd.normal_w
        # ]

        start_time = time.time()
        # decoded_primitives, decoded_sh, decoded_r, decoded_s, decoded_o, bpp,
        #         center_points, decoded_offsets, _, bpp_clr, bpp_hyper, decoded_n = self.model(color_sparse)
        decoded_primitives = points_collate
        decoded_sh_dc = [
                RGB2SH(color_feats[i]).unsqueeze(-2) for i in range(len(color_feats))
            ]
        sh_deg = 1
        pseudo_sh_dim = (2 ** (sh_deg + 1)) * 3
        decoded_sh_ac = [
            torch.zeros((color_feats[i].shape[0], pseudo_sh_dim, 3), device=color_feats[i].device) for i in range(len(color_feats))
        ]
        decoded_sh = [
            torch.cat([decoded_sh_dc[i], decoded_sh_ac[i]], dim=1) for i in range(len(decoded_sh_dc))
        ]
        # decoded_n = [
        #     normal_w_feats[i] for i in range(len(normal_w_feats))
        # ]
        decoded_n = None

        model_time = time.time() - start_time
        if self.voxelized:
            ray_space_points_sparse_recon = [
                    pcgc_rescale(
                        decoded_primitives[i].float(), self.offset, self.scale_factor, device=self.device)
                        for i in range(len(decoded_primitives))
            ]
        else:
            ray_space_points_sparse_recon = [
                    decoded_primitives[i].float() for i in range(len(decoded_primitives))
            ]
        decoded_r = [
            self.default_quaternion.expand(color_feats[i].shape[0], 4) for i in range(len(color_feats))
        ]
        if self.voxelized:
            scale_norm_factor = self.scale_factor
        else:
            scale_norm_factor = 1.
        decoded_s = [
            torch.ones_like(color_feats[i][:, 0:3], device=color_feats[i].device) * sigma / scale_norm_factor for i in range(len(color_feats))
        ]

        decoded_o = [torch.ones_like(color_feats[i][:, 0:1], device=color_feats[i].device) for i in range(len(color_feats))]
        # Render
        batchsize = len(ray_space_points_sparse_recon)

        gt_camera= cam.to(self.device)
        num_q = gt_camera.H_c2w.shape[1]
        camera_chunks = gt_camera.chunk(batchsize, dim=0)

        rendered_recons = []
        radii_recons = []
        bg = torch.zeros(3).to(device=self.device) + background_color

        # Render RGB
        primitive_dict = {
            'decoded_o': decoded_o,
            'decoded_s': decoded_s,
            'decoded_r': decoded_r,
            'decoded_sh': decoded_sh,
            'colors_precomp': None,
        }
        rendered_recons = self._rasterize(
            batchsize,
            ray_space_points_sparse_recon,
            primitive_dict,
            camera_chunks,
            num_q,
            bg,
            cam.height_px,
            cam.width_px,
            super_sample_rate,
            fov,
        )
        rgb_time = time.time() - start_time - model_time
        print('model time: %.3f sec, rgb time: %.3f sec' % (model_time, rgb_time), flush=True)
        
        # Render world xyz
        primitive_dict = {
            'decoded_o': decoded_o,
            'decoded_s': decoded_s,
            'decoded_r': decoded_r,
            'decoded_sh': None,
            'colors_precomp': ray_space_points_sparse_recon,
        }
        rendered_xyz = self._rasterize(
            batchsize,
            ray_space_points_sparse_recon,
            primitive_dict,
            camera_chunks,
            num_q,
            bg,
            cam.height_px,
            cam.width_px,
            super_sample_rate,
            fov,
        )

        # Render hitmap
        primitive_dict = {
            'decoded_o': decoded_o,
            'decoded_s': decoded_s,
            'decoded_r': decoded_r,
            'decoded_sh': None,
            'colors_precomp': [torch.ones_like(ray_space_points_sparse_recon[i]) for i in range(len(ray_space_points_sparse_recon))],
        }
        rendered_hitmap = self._rasterize(
            batchsize,
            ray_space_points_sparse_recon,
            primitive_dict,
            camera_chunks,
            num_q,
            bg,
            cam.height_px,
            cam.width_px,
            super_sample_rate,
            fov,
        )

        # Render normal (optional)
        if decoded_n is not None:
            decoded_consistent_n = []
            assert not consistent_normal
            
            decoded_consistent_n = decoded_n
                
            primitive_dict = {
                'decoded_o': decoded_o,
                'decoded_s': decoded_s,
                'decoded_r': decoded_r,
                'decoded_sh': None,
                'colors_precomp': decoded_consistent_n,
            }
            # process decoded_n
            rendered_normals = self._rasterize(
                batchsize,
                ray_space_points_sparse_recon,
                primitive_dict,
                camera_chunks,
                num_q,
                bg,
                cam.height_px,
                cam.width_px,
                super_sample_rate,
                fov,
                normalize_camera_normal=True,
            )
        else:
            rendered_normals = None
        
        
        ret_dict = {
            'rgb': rendered_recons,
            'normal': rendered_normals,
            'xyz_w': rendered_xyz,
            'hitmap': rendered_hitmap,
        }
        
        if point_light is not None:
            lighted_maps = [ret_dict['rgb'] * point_light['light_coeff'][0]]
            for i in range(len(point_light['xyz_w'])):
                light_dir = ret_dict['xyz_w'] - point_light['xyz_w'][i].to(self.device)
                light_dir = light_dir / torch.norm(light_dir, dim=-1, keepdim=True)
                cos_theta = torch.sum(light_dir * ret_dict['normal'], dim=-1, keepdim=True)
                cos_theta = torch.clamp(cos_theta, min=0)
                # cos_theta = torch.abs(cos_theta)
                lighted_i = point_light['color'][i].to(self.device) * cos_theta * ret_dict['hitmap'] * ret_dict['rgb'] * point_light['light_coeff'][i+1]
                lighted_maps.append(lighted_i)
            ret_dict['shaded'] = torch.sum(torch.stack(lighted_maps, dim=0), dim=0)
        
        return ret_dict 

def get_gt(pth,cam):
    mesh = Mesh(pth, scale=1.)
    output_ray = cam.generate_camera_rays(
            subsample=1,
            offsets='center',
        )  # (b=1, q, ho, wo)
    gt_dict = mesh.get_ray_intersection(
        ray=output_ray,
    )
    return gt_dict

def compare_psnr(gt_rgb, rgb, device):
    gt_rgb, rgb = gt_rgb.to(device), rgb.to(device)
    diff = (gt_rgb - rgb) * 255
    mse = torch.mean(diff ** 2,dim=(-1,-2,-3))
    mse = mse.detach().cpu().numpy()
    psnr = 20 * np.log10(255*np.ones(mse.shape)) - 10 * np.log10(mse)
    return np.mean(psnr)
