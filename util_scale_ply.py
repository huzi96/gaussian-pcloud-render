import open3d as o3d
import numpy as np
import os
import sys
import argparse
import glob

def pcgc_rescale(input_xyz, factor=256):
    #input_xyz = input_xyz * (1 / factor)
    res = (input_xyz - 512) / factor
    return res

def pcgc_scale(input_xyz, factor=256):
    #res = input_xyz + 1.0
    res = res*factor +512 
    return res

parser = argparse.ArgumentParser()
parser.add_argument('--input_ply', type=str, default='pcd_0.ply')
parser.add_argument('--output_ply', type=str, default='scale_pcd_0.ply')
parser.add_argument('--factor', type=int, default=256)
def main(args):
    fn = args.input_ply
    print("Processing: ", fn)
    pcd = o3d.io.read_point_cloud(fn)
    xyz = np.asarray(pcd.points)
    xyz = pcgc_scale(xyz, args.factor)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(args.output_ply, pcd, write_ascii=True)
    print("Done: ", args.output_ply)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

