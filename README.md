# 3D Gaussian based Point Cloud Renderer
### <a href="https://huzi96.github.io/">Yueyu Hu</a>, Ran Gong, <a href="https://www.immersivecomputinglab.org">Qi Sun</a>, <a href="https://wp.nyu.edu/videolab/">Yao Wang</a>.
Code repo for paper "Low Latency Point Cloud Rendering with Learned Splatting", CVPR Workshop (AIS: Vision, Graphics and AI for Streaming), 2024.

<a href="https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/papers/Hu_Low_Latency_Point_Cloud_Rendering_with_Learned_Splatting_CVPRW_2024_paper.pdf">[PDF]</a> <a href="https://openaccess.thecvf.com/content/CVPR2024W/AI4Streaming/supplemental/Hu_Low_Latency_Point_CVPRW_2024_supplemental.pdf">[supp]</a> <a href="https://ai4streaming-workshop.github.io/">[Workshop]</a>

#### Related work:
Yueyu Hu, Ran Gong, Yao Wang. "Bits-to-Photon: End-to-End Learned Scalable Point Cloud Compression for Direct Rendering", <a href="https://arxiv.org/abs/2406.05915">arXiv:2406.05915</a>, 2024.

This development of this repo is largely helped by and depending on the following open-source projects:

**Pointersect**: https://github.com/apple/ml-pointersect

**3D Gaussian Splatting**: https://github.com/graphdeco-inria/gaussian-splatting

## Dependencies
### PyTorch
The code is tested with PyTorch == 1.12.1 and CUDA 11.3, on NVIDIA RTX 4080 Super. Install PyTorch with,

```conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch```

### MinkowskiEngine

Please follow https://github.com/NVIDIA/MinkowskiEngine to install MinkowskiEngine. The following command might simply work,

```sudo apt install build-essential python3-dev libopenblas-dev```

```pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps```

### Others

```pip install imageio open3d==0.16.0 opencv-python torch_scatter xatlas scikit-image scipy pyexr pytorch_msssim lpips```

### Install Diff Gaussian Rasterization Package
```
cd diff-gaussian-rasterization
MAKEFLAGS="-j8" pip install .
```

## Run example

### Example 1: Quantized (200K)
```python simple_benchmark.py pcrender --dataset_root ./example/THuman-256 --scale_factor 256 --fov 45 --voxelized --id_list 0519```

### Example 2: Non-quantized (800K)
```python simple_benchmark.py pcrender --dataset_root ./example/THuman-800K --scale_factor 448 --fov 45 --id_list 0519```

## Test with more data samples with a mesh dataset

We provide as script ```sample_point_cloud_from_mesh.py``` that samples point clouds from meshes for testing. Please refer to the help message by ```python sample_point_cloud_from_mesh.py -h``` for usage.