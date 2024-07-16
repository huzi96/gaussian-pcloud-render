#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# The file implements a dataloader which generates a dataset from a mesh.

import time
import traceback
import typing as T

import imageio
import numpy as np
import torch

import structures
from structures import CameraTrajectory
from tqdm import tqdm
import math

class MeshDataset:

    def __init__(
            self,
            mesh: structures.Mesh,
            pcd: structures.PointCloud = None,
            kd_tree = None,
            n_target_imgs: int = 3,
            n_imgs: int = 3,
            total: int = 10000,
            input_camera_setting: T.Dict[str, T.Any] = None,
            input_camera_trajectory_params: T.Dict[str, T.Any] = None,
            output_camera_setting: T.Dict[str, T.Any] = None,
            output_camera_trajectory_params: T.Dict[str, T.Any] = None,
            rng_seed: T.Union[int, None] = 0,
            render_method: str = 'ray_cast',
            texture_filenames: T.List[str] = None,
            texture_crop_method: str = 'full',  # 'random', number
            min_subsample: int = 1,
            max_subsample: int = 1,
    ):
        """
        Args:
            mesh:
                class Mesh containing cleaned and scaled mesh and ray_casting scene.
            n_target_imgs:
                number of target images for a pcd sample (think of it as random rays)
            n_imgs:
                number of input images to create an input point cloud
            total:
                number of (pcd_sample, target) pairs (dataset size)
            input_camera_settings:
                width_px: number of pixel in the x direction
                height_px: number of pixel in the x direction
                fov:  field ov view in degree
                ray_offsets:  'center', 'random'
            input_camera_trajectory_params:
                mode: mode of the camera trajectory
                min_r: minimum distance from camera to the mesh origin. None: min_r will be set to the min aabb size
                max_r: maximum distance from camera to the mesh origin.
                max_angle: spread of the images in viewing direction (in degree)
                local_max_angle: small local rotation of each images (in degree)
            output_camera_settings:
                see `input_camera_settings`
            output_camera_trajectory_params:
                see `input_camera_trajectory_params`
                ray_perturbation_angle:  small angle to perturb target ray direction (in degree). If None: use fov / width_px.
            rng_seed:
                random seed to use
            render_method:
                'rasterization'
                'ray_cast'
            texture_filenames:
                list of texture images to randomly use to replace the original texture
            texture_crop_method:
                'full': use the full image
                int: random crop
            min_subsample:
                to randomly generate subsampled input rgbd images.
            max_subsample:
                to randomly generate subsampled input rgbd images.
        """

        self.mesh = mesh
        self.pcd = pcd
        # self.kd_tree = kd_tree
        self.n_target_imgs = n_target_imgs
        self.n_imgs = n_imgs
        self.total = total
        self.render_method = render_method
        self.texture_filenames = texture_filenames
        self.texture_crop_method = texture_crop_method
        self.min_subsample = max(1, min_subsample)
        self.max_subsample = max_subsample
        if self.max_subsample is None:
            self.max_subsample = self.min_subsample
        else:
            self.max_subsample = max(1, self.max_subsample)

        if rng_seed is not None:
            self.rng = np.random.RandomState(seed=rng_seed)
        else:
            self.rng = np.random

        # get mesh size
        self.min_bounds = self.mesh.mesh.get_min_bound()  # (3,)  xyz  top left
        self.max_bounds = self.mesh.mesh.get_max_bound()  # (3,)  xyz  bottom right

        # camera settings
        self._set_camera_defaults()
        if input_camera_setting is not None:
            self.input_camera_setting.update(input_camera_setting)
        if input_camera_trajectory_params is not None:
            self.input_camera_trajectory_params.update(input_camera_trajectory_params)
        if output_camera_setting is not None:
            self.output_camera_setting.update(output_camera_setting)
        if output_camera_trajectory_params is not None:
            self.output_camera_trajectory_params.update(output_camera_trajectory_params)

        if self.n_target_imgs > 0:
            output_cam_trajectory = structures.CameraTrajectory(
                mode=self.output_camera_trajectory_params['mode'],
                n_imgs=self.n_target_imgs,
                total=self.total,
                rng_seed=self.rng,
                params=self.output_camera_trajectory_params,
            )
            self.output_cameras = output_cam_trajectory.get_camera(
                fov=self.output_camera_setting['fov'],
                width_px=self.output_camera_setting['width_px'],
                height_px=self.output_camera_setting['height_px'],
                device=torch.device('cpu'),
            )
        else:
            self.output_cameras = None

        # create texture replacement index
        if self.texture_filenames is not None:
            textures = self.mesh.mesh.textures  # list of textures
            num_textures = max(1, len(textures))
            self.texture_idxs = self.rng.randint(
                len(self.texture_filenames),
                size=(self.total, num_textures),
            )  # (total, n_texture)
        else:
            self.texture_idxs = None

    def get_all_num_pixels(self) -> T.List[int]:
        """Return the number of pixels in each image."""
        n_pxs = []
        for i in range(self.total):
            n_px = self.pcd.get_num_points()
            n_pxs.append(n_px)
        return n_pxs

    def _set_camera_defaults(self):
        """Set the defaults for attributes."""

        self.input_camera_trajectory_params: T.Dict[str, T.Any] = dict(
            mode='random',
            min_r=None,
            max_r=None,
            max_angle=30.,
            num_circle=4,
            local_max_angle=3.,
            r_freq=1,
            max_translate_ratio=2.,
            rand_r=0.,
        )
        self.output_camera_trajectory_params: T.Dict[str, T.Any] = dict(
            mode='random',
            min_r=None,
            max_r=None,
            max_angle=30.,
            num_circle=4,
            r_freq=1,
            max_translate_ratio=2.,
            local_max_angle=3.,
            perturb_shift=None,
            perturb_angle=None,
            rand_r=0.,
        )
        self.input_camera_setting: T.Dict[str, T.Any] = dict(
            fov=60,
            width_px=100,
            height_px=100,
        )
        self.output_camera_setting: T.Dict[str, T.Any] = dict(
            fov=60,
            width_px=20,
            height_px=20,
            ray_offsets='center',
        )

        if self.input_camera_trajectory_params['min_r'] is None:
            self.input_camera_trajectory_params['min_r'] = \
                max(np.max(np.abs(self.min_bounds)), np.max(np.abs(self.max_bounds)))
        if self.input_camera_trajectory_params['max_r'] is None:
            self.input_camera_trajectory_params['max_r'] = \
                self.input_camera_trajectory_params['max_translate_ratio'] \
                * self.input_camera_trajectory_params['min_r']

        if self.output_camera_trajectory_params['min_r'] is None:
            self.output_camera_trajectory_params['min_r'] = self.input_camera_trajectory_params['min_r']
        if self.output_camera_trajectory_params['max_r'] is None:
            self.output_camera_trajectory_params['max_r'] = \
                self.output_camera_trajectory_params['max_translate_ratio'] \
                * self.output_camera_trajectory_params['min_r']

    def __len__(self):
        return self.total

    def __getitem__(self, i) -> T.Dict[str, T.Any]:
        """
        Returns:
            point_cloud:
                PointCloud, (b=1, n),  n: number of points
            ray:
                Ray, (b=1, q=n_target_img, ho, wo)  target rays
            ray_gt_dict:
                ray_rgbs: (b=1, q=n_target_img, ho, wo, 3)
                ray_ts: (b=1, q=n_target_img, ho, wo)
                surface_normals_w: (b=1, q=n_target_img, ho, wo, 3)
                hit_map: (b=1, q=n_target_img, ho, wo)  1 if hit a surface, 0 otherwise
        """

        # read texture images
        if self.texture_idxs is not None:
            texture_imgs = []
            for tidx in range(len(self.texture_idxs[i])):
                timeout_sec = 5
                try:
                    img = imageio.v3.imread(self.texture_filenames[self.texture_idxs[i, tidx]])
                except Exception as e:
                    print(
                        f"reading image {self.texture_filenames[self.texture_idxs[i, tidx]]}, failed")
                    img = np.ones((500, 500, 3))

                if len(img.shape) == 2:
                    img = np.stack((img, img, img), axis=-1)

                if img.shape[-1] > 3:
                    img = img[..., :3]

                # crop image
                h, w = img.shape[:2]
                if self.texture_crop_method == 'full':
                    s = min(h, w)
                elif isinstance(self.texture_crop_method, str):
                    # {min_s}-{max_s}
                    min_s, max_s = self.texture_crop_method.split('-', 1)
                    max_s = min(w, min(h, int(max_s)))
                    min_s = int(min_s)
                    min_s = max(1, min_s)
                    max_s = max(1, max_s)
                    min_s = min(min_s, max_s)
                    s = self.rng.randint(low=min_s, high=max_s + 1)
                elif isinstance(self.texture_crop_method, int):
                    max_s = min(h, w)
                    min_s = min(max_s, self.texture_crop_method)
                    s = self.rng.randint(low=min_s, high=max_s + 1)
                else:
                    raise NotImplementedError

                w_from = self.rng.randint(w - s + 1)
                h_from = self.rng.randint(h - s + 1)
                img = img[max(0, h_from):min(h, h_from + s), max(0, w_from):min(w, w_from + s)]
                texture_imgs.append(img)

            self.mesh.replace_texture(texture_imgs)

        # generate ground-truth output for ray
        output_ray = self.output_cameras[i].generate_camera_rays(
            subsample=1,
            offsets=self.output_camera_setting['ray_offsets'],
        )  # (b=1, q, ho, wo)

        # randomly perturb the ray
        # output_ray.random_perturb_direction(
        #     shift=self.output_camera_trajectory_params['perturb_shift'],
        #     angle=self.output_camera_trajectory_params['perturb_angle'],
        # )

        gt_dict = self.mesh.get_ray_intersection(
            ray=output_ray,
        )
        # ray_rgbs: (b=1, q, ho, wo, 3)
        # ray_ts: (b=1, q, ho, wo)
        # surface_normals_w: (b=1, q, ho, wo, 3)
        # hit_map: (b=1, q, ho, wo) 1: hit or 0: not hit

        out_dict = dict(
            pcd = self.pcd,
            ray=output_ray,
            ray_gt_dict=gt_dict,
            cam = self.output_cameras[i],
            # kd_tree = self.kd_tree
        )

        return out_dict

class MeshDatasetMultiPCD:

    def __init__(
            self,
            mesh: structures.Mesh,
            pcd: structures.PointCloud = None,
            kd_tree = None,
            n_target_imgs: int = 3,
            n_imgs: int = 3,
            total: int = 10000,
            input_camera_setting: T.Dict[str, T.Any] = None,
            input_camera_trajectory_params: T.Dict[str, T.Any] = None,
            output_camera_setting: T.Dict[str, T.Any] = None,
            output_camera_trajectory_params: T.Dict[str, T.Any] = None,
            rng_seed: T.Union[int, None] = 0,
            render_method: str = 'ray_cast',
            texture_filenames: T.List[str] = None,
            texture_crop_method: str = 'full',  # 'random', number
            min_subsample: int = 1,
            max_subsample: int = 1,
            num_pcd: int = 1,
    ):
        """
        Args:
            mesh:
                class Mesh containing cleaned and scaled mesh and ray_casting scene.
            n_target_imgs:
                number of target images for a pcd sample (think of it as random rays)
            n_imgs:
                number of input images to create an input point cloud
            total:
                number of (pcd_sample, target) pairs (dataset size)
            input_camera_settings:
                width_px: number of pixel in the x direction
                height_px: number of pixel in the x direction
                fov:  field ov view in degree
                ray_offsets:  'center', 'random'
            input_camera_trajectory_params:
                mode: mode of the camera trajectory
                min_r: minimum distance from camera to the mesh origin. None: min_r will be set to the min aabb size
                max_r: maximum distance from camera to the mesh origin.
                max_angle: spread of the images in viewing direction (in degree)
                local_max_angle: small local rotation of each images (in degree)
            output_camera_settings:
                see `input_camera_settings`
            output_camera_trajectory_params:
                see `input_camera_trajectory_params`
                ray_perturbation_angle:  small angle to perturb target ray direction (in degree). If None: use fov / width_px.
            rng_seed:
                random seed to use
            render_method:
                'rasterization'
                'ray_cast'
            texture_filenames:
                list of texture images to randomly use to replace the original texture
            texture_crop_method:
                'full': use the full image
                int: random crop
            min_subsample:
                to randomly generate subsampled input rgbd images.
            max_subsample:
                to randomly generate subsampled input rgbd images.
        """

        self.mesh = mesh
        self.pcd = pcd
        self.num_pcd = num_pcd
        assert num_pcd == len(pcd)
        # self.kd_tree = kd_tree
        self.n_target_imgs = n_target_imgs
        self.n_imgs = n_imgs
        self.total = total
        self.render_method = render_method
        self.texture_filenames = texture_filenames
        self.texture_crop_method = texture_crop_method
        self.min_subsample = max(1, min_subsample)
        self.max_subsample = max_subsample
        if self.max_subsample is None:
            self.max_subsample = self.min_subsample
        else:
            self.max_subsample = max(1, self.max_subsample)

        if rng_seed is not None:
            self.rng = np.random.RandomState(seed=rng_seed)
        else:
            self.rng = np.random

        # get mesh size
        self.min_bounds = self.mesh.mesh.get_min_bound()  # (3,)  xyz  top left
        self.max_bounds = self.mesh.mesh.get_max_bound()  # (3,)  xyz  bottom right

        # camera settings
        self._set_camera_defaults()
        if input_camera_setting is not None:
            self.input_camera_setting.update(input_camera_setting)
        if input_camera_trajectory_params is not None:
            self.input_camera_trajectory_params.update(input_camera_trajectory_params)
        if output_camera_setting is not None:
            self.output_camera_setting.update(output_camera_setting)
        if output_camera_trajectory_params is not None:
            self.output_camera_trajectory_params.update(output_camera_trajectory_params)

        if self.n_target_imgs > 0:
            output_cam_trajectory = structures.CameraTrajectory(
                mode=self.output_camera_trajectory_params['mode'],
                n_imgs=self.n_target_imgs,
                total=self.total,
                rng_seed=self.rng,
                params=self.output_camera_trajectory_params,
            )
            self.output_cameras = output_cam_trajectory.get_camera(
                fov=self.output_camera_setting['fov'],
                width_px=self.output_camera_setting['width_px'],
                height_px=self.output_camera_setting['height_px'],
                device=torch.device('cpu'),
            )
        else:
            self.output_cameras = None

        # create texture replacement index
        if self.texture_filenames is not None:
            textures = self.mesh.mesh.textures  # list of textures
            num_textures = max(1, len(textures))
            self.texture_idxs = self.rng.randint(
                len(self.texture_filenames),
                size=(self.total, num_textures),
            )  # (total, n_texture)
        else:
            self.texture_idxs = None

    def get_all_num_pixels(self) -> T.List[int]:
        """Return the number of pixels in each image."""
        n_pxs = []
        for i in range(self.total):
            n_px = self.pcd.get_num_points()
            n_pxs.append(n_px)
        return n_pxs

    def _set_camera_defaults(self):
        """Set the defaults for attributes."""

        self.input_camera_trajectory_params: T.Dict[str, T.Any] = dict(
            mode='random',
            min_r=None,
            max_r=None,
            max_angle=30.,
            num_circle=4,
            local_max_angle=3.,
            r_freq=1,
            max_translate_ratio=2.,
            rand_r=0.,
        )
        self.output_camera_trajectory_params: T.Dict[str, T.Any] = dict(
            mode='random',
            min_r=None,
            max_r=None,
            max_angle=30.,
            num_circle=4,
            r_freq=1,
            max_translate_ratio=2.,
            local_max_angle=3.,
            perturb_shift=None,
            perturb_angle=None,
            rand_r=0.,
        )
        self.input_camera_setting: T.Dict[str, T.Any] = dict(
            fov=60,
            width_px=100,
            height_px=100,
        )
        self.output_camera_setting: T.Dict[str, T.Any] = dict(
            fov=60,
            width_px=20,
            height_px=20,
            ray_offsets='center',
        )

        if self.input_camera_trajectory_params['min_r'] is None:
            self.input_camera_trajectory_params['min_r'] = \
                max(np.max(np.abs(self.min_bounds)), np.max(np.abs(self.max_bounds)))
        if self.input_camera_trajectory_params['max_r'] is None:
            self.input_camera_trajectory_params['max_r'] = \
                self.input_camera_trajectory_params['max_translate_ratio'] \
                * self.input_camera_trajectory_params['min_r']

        if self.output_camera_trajectory_params['min_r'] is None:
            self.output_camera_trajectory_params['min_r'] = self.input_camera_trajectory_params['min_r']
        if self.output_camera_trajectory_params['max_r'] is None:
            self.output_camera_trajectory_params['max_r'] = \
                self.output_camera_trajectory_params['max_translate_ratio'] \
                * self.output_camera_trajectory_params['min_r']

    def __len__(self):
        return self.total

    def __getitem__(self, i) -> T.Dict[str, T.Any]:
        """
        Returns:
            point_cloud:
                PointCloud, (b=1, n),  n: number of points
            ray:
                Ray, (b=1, q=n_target_img, ho, wo)  target rays
            ray_gt_dict:
                ray_rgbs: (b=1, q=n_target_img, ho, wo, 3)
                ray_ts: (b=1, q=n_target_img, ho, wo)
                surface_normals_w: (b=1, q=n_target_img, ho, wo, 3)
                hit_map: (b=1, q=n_target_img, ho, wo)  1 if hit a surface, 0 otherwise
        """

        # read texture images
        if self.texture_idxs is not None:
            texture_imgs = []
            for tidx in range(len(self.texture_idxs[i])):
                timeout_sec = 5
                try:
                    img = imageio.v3.imread(self.texture_filenames[self.texture_idxs[i, tidx]])
                except Exception as e:
                    print(
                        f"reading image {self.texture_filenames[self.texture_idxs[i, tidx]]}, failed")
                    img = np.ones((500, 500, 3))

                if len(img.shape) == 2:
                    img = np.stack((img, img, img), axis=-1)

                if img.shape[-1] > 3:
                    img = img[..., :3]

                # crop image
                h, w = img.shape[:2]
                if self.texture_crop_method == 'full':
                    s = min(h, w)
                elif isinstance(self.texture_crop_method, str):
                    # {min_s}-{max_s}
                    min_s, max_s = self.texture_crop_method.split('-', 1)
                    max_s = min(w, min(h, int(max_s)))
                    min_s = int(min_s)
                    min_s = max(1, min_s)
                    max_s = max(1, max_s)
                    min_s = min(min_s, max_s)
                    s = self.rng.randint(low=min_s, high=max_s + 1)
                elif isinstance(self.texture_crop_method, int):
                    max_s = min(h, w)
                    min_s = min(max_s, self.texture_crop_method)
                    s = self.rng.randint(low=min_s, high=max_s + 1)
                else:
                    raise NotImplementedError

                w_from = self.rng.randint(w - s + 1)
                h_from = self.rng.randint(h - s + 1)
                img = img[max(0, h_from):min(h, h_from + s), max(0, w_from):min(w, w_from + s)]
                texture_imgs.append(img)

            self.mesh.replace_texture(texture_imgs)

        # generate ground-truth output for ray
        output_ray = self.output_cameras[i].generate_camera_rays(
            subsample=1,
            offsets=self.output_camera_setting['ray_offsets'],
        )  # (b=1, q, ho, wo)

        # randomly perturb the ray
        # output_ray.random_perturb_direction(
        #     shift=self.output_camera_trajectory_params['perturb_shift'],
        #     angle=self.output_camera_trajectory_params['perturb_angle'],
        # )

        gt_dict = self.mesh.get_ray_intersection(
            ray=output_ray,
        )
        # ray_rgbs: (b=1, q, ho, wo, 3)
        # ray_ts: (b=1, q, ho, wo)
        # surface_normals_w: (b=1, q, ho, wo, 3)
        # hit_map: (b=1, q, ho, wo) 1: hit or 0: not hit

        rnd_index = self.rng.randint(0, self.num_pcd)
        out_dict = dict(
            pcd = self.pcd[rnd_index],
            ray=output_ray,
            ray_gt_dict=gt_dict,
            cam = self.output_cameras[i],
            # kd_tree = self.kd_tree
        )

        return out_dict

class MeshRGBDDataset(MeshDataset):
    def __init__(
            self,
            mesh: structures.Mesh,
            pcd: structures.PointCloud = None,
            kd_tree = None,
            n_target_imgs: int = 3,
            n_imgs: int = 3,
            total: int = 10000,
            input_camera_setting: T.Dict[str, T.Any] = None,
            input_camera_trajectory_params: T.Dict[str, T.Any] = None,
            output_camera_setting: T.Dict[str, T.Any] = None,
            output_camera_trajectory_params: T.Dict[str, T.Any] = None,
            rng_seed: T.Union[int, None] = 0,
            render_method: str = 'ray_cast',
            texture_filenames: T.List[str] = None,
            texture_crop_method: str = 'full',  # 'random', number
            min_subsample: int = 1,
            max_subsample: int = 1,
            min_rnd_num_points: int = 100000,
            max_rnd_num_points: int = 1000000,
            perturb_camera: bool = False,
            fov: float = 60.,
            perturb_camera_origin_rate: float = 0.
    ):
        super().__init__(
            mesh=mesh,
            pcd=pcd,
            kd_tree=kd_tree,
            n_target_imgs=n_target_imgs,
            n_imgs=n_imgs,
            total=total,
            input_camera_setting=input_camera_setting,
            input_camera_trajectory_params=input_camera_trajectory_params,
            output_camera_setting=output_camera_setting,
            output_camera_trajectory_params=output_camera_trajectory_params,
            rng_seed=rng_seed,
            render_method=render_method,
            texture_filenames=texture_filenames,
            texture_crop_method=texture_crop_method,
            min_subsample=min_subsample,
            max_subsample=max_subsample,
        )
        self.min_rnd_num_points = min_rnd_num_points
        self.max_rnd_num_points = max_rnd_num_points
        self.perturb_camera = perturb_camera
        self.fov = fov
        self.perturb_camera_origin_rate = perturb_camera_origin_rate
        if self.perturb_camera:
            # TODO: implement perturb_camera
            raise NotImplementedError('perturb_camera is not implemented yet')
    
    def __getitem__(self, i) -> T.Dict[str, T.Any]:
        num_points = np.random.randint(self.min_rnd_num_points, self.max_rnd_num_points)
        width_px = 128
        height_px = 128
        n_imgs = max(3, num_points // (width_px * height_px))
        n_pixels_per_img = num_points / n_imgs
        width_px = max(2, math.floor(n_pixels_per_img / (width_px * height_px) * width_px))
        width_px = max(2, width_px - (width_px % 2))
        height_px = max(2, math.floor(n_pixels_per_img / width_px))
        height_px = max(2, height_px - (height_px % 2))

        # get mesh scale and center
        cs = self.mesh.center_w
        s = self.mesh.scale

        # create uniformly placed camera
        camera = CameraTrajectory(
            mode='random',
            n_imgs=n_imgs,
            total=1,
            params=dict(
                max_angle=180,
                min_r=2 * s,
                max_r=2 * s + 1.e-9,
                origin_w=cs.tolist(),
                method='LatinHypercube',
            ),
            dtype=np.float32,
        ).get_camera(
            fov=self.fov,
            width_px=width_px,
            height_px=height_px,
            device=torch.device('cpu'),
        )
        rgbd_image = self.mesh.get_rgbd_image(
            camera=camera,
            render_method='ray_cast',  # 'ray_cast',
            device=torch.device('cpu'),
        )
        point_cloud = rgbd_image.get_pcd(
            perturb_camera_origin_rate=self.perturb_camera_origin_rate)
    
        # generate ground-truth output for ray
        output_ray = self.output_cameras[i].generate_camera_rays(
            subsample=1,
            offsets=self.output_camera_setting['ray_offsets'],
        )  # (b=1, q, ho, wo)

        gt_dict = self.mesh.get_ray_intersection(
            ray=output_ray,
        )

        out_dict = dict(
            pcd = point_cloud,
            ray=output_ray,
            ray_gt_dict=gt_dict,
            cam = self.output_cameras[i],
            # kd_tree = self.kd_tree
        )
        return out_dict

class MeshDatasetCollate:

    def __call__(
            self,
            input_list: T.List[T.Dict[str, T.Any]],
    ):
        """
        Returns:
            input_rgbd_images:
                RGBDImage, (b, q, h, w),
            ray:
                Ray, (b, q=n_target_img, ho, wo)  target rays
            ray_gt_dict:
                ray_rgbs: (b, q=n_target_img, ho, wo, 3)
                ray_ts: (b, q=n_target_img, ho, wo)
                surface_normals_w: (b, q=n_target_img, ho, wo, 3)
                hit_map: (b, q=n_target_img, ho, wo)  1 if hit a surface, 0 otherwise
        """

        # input pcd
        pcd = structures.PointCloud.cat([p['pcd'] for p in input_list], dim=0)  # (b, q, h, w)

        # ray
        ray = structures.Ray.cat([p['ray'] for p in input_list], dim=0)  # (b, q, h, w)

        # ray_gt_dict
        ray_gt_dict = dict()
        for key in ['ray_rgbs', 'ray_ts', 'surface_normals_w', 'hit_map']:
            arr = torch.cat([p['ray_gt_dict'][key] for p in input_list], dim=0)
            ray_gt_dict[key] = arr

        cam = structures.Camera.cat([p['cam'] for p in input_list], dim=0) 

        return dict(
            ray=ray,
            ray_gt_dict=ray_gt_dict,
            cam = cam,
            pcd = pcd,
            # kd_tree = [p['kd_tree'] for p in input_list]
        )


class MeshConcatDataset(torch.utils.data.Dataset):
    r"""Dataset as a concatenation of multiple MeshDatasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    def __init__(
            self,
            datasets: T.Iterable,
            max_retry=30,
            wait_sec=5,
    ):

        self.datasets = list(datasets)
        assert len(self.datasets) > 0
        self.concat_dataset = torch.utils.data.ConcatDataset(self.datasets)
        self.max_retry = max_retry
        self.wait_sec = wait_sec

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        d = None
        for retry in range(self.max_retry):
            try:
                d = self.concat_dataset[idx]
                if d is not None:
                    return d
            except:
                traceback.print_exc()
                time.sleep(self.wait_sec)
        return d

    def get_all_num_pixels(self):
        all_seq_lens = []
        for dset in self.datasets:
            seq_lens = dset.get_all_num_pixels()
            if seq_lens is None:
                return None
            else:
                all_seq_lens += seq_lens
        return all_seq_lens
