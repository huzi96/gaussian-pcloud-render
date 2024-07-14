# Sample point cloud from meshes using multiprocessing
import os
import argparse
from structures import Mesh
from tqdm import tqdm
import multiprocessing as mp
import glob

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default='./dataset/THuman/', help='Path to the dataset containing folders of meshes in formats ASSET_ID/ASSET_ID.obj.')
    parser.add_argument("--numpoints", type=int, default=1000000, help='Number of points to sample from each mesh. Resulting point cloud will have fewer points if quantization is enabled.')
    parser.add_argument("--range", type=str, default='0-20', help='Range of folder indices to process.')
    parser.add_argument("--method", type=str, default='uniform_quantized', choices=['uniform_quantized', 'uniform', 'poisson_disk'], help='Method to sample points from mesh. By default, points are sampled uniformly and quantized to the grid. Generally poisson_disk gives more representative point clouds.')
    args = parser.parse_args()
    return args

def sample_mesh(mesh_fn, num_points, savepth, method):
    print('[Worker] Loading mesh from %s' % mesh_fn)
    mesh = Mesh(mesh=mesh_fn)
    print('[Worker] Sampling mesh to %d points' % num_points)
    pcd = mesh.sample_point_cloud(
        num_points=num_points,
        method=method
        )["point_cloud"]
    print('[Worker] Saving point cloud to %s' % savepth)
    pcd.save(
        output_dir=savepth,
        overwrite=True,
        save_ply=True,
        save_pt=False,
        prefix=mesh_fn.split('/')[-1].split('.')[-2]+'_')
    
def error_callback(e):
    print(e)

if __name__ == '__main__':
    print(mp.set_start_method('forkserver'))
    args = parse_args()
    data_root = args.dataset
    ids = sorted(os.listdir(data_root))
    pool = mp.Pool(processes=8)
    m = mp.Manager()
    l = m.Lock()
    fr, to = args.range.split('-')
    fr = int(fr)
    to = int(to)
    for id in tqdm(ids[fr:to]):
        savepth = os.path.join(data_root, id)
        obj_files = sorted(glob.glob(os.path.join(savepth, '*.obj')))
        for mesh_fn in obj_files:
            pool.apply_async(
                sample_mesh, args=(mesh_fn, args.numpoints, savepth, args.method), 
                error_callback=error_callback)
    pool.close()
    pool.join()
