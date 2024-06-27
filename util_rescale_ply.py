import open3d as o3d
import numpy as np
import os
import sys
import argparse
import glob

def pcgc_rescale(input_xyz, factor=256, offset=512):
    #input_xyz = input_xyz * (1 / factor)
    res = (input_xyz - offset) / factor
    return res

def pcgc_scale(input_xyz, factor=256, offset=512):
    #res = input_xyz + 1.0
    res = res*factor +offset 
    return res

parser = argparse.ArgumentParser()
parser.add_argument('--input_ply', type=str, default='longdress_vox10_1300_dec.ply')
parser.add_argument('--output_ply', type=str, default='pcd_0.ply')
parser.add_argument('--factor', type=int, default=256)
parser.add_argument('--input_offset', type=str, default='0,0,0')
parser.add_argument('--offset', type=int, default=512)
def main(args):
    fn = args.input_ply
    print("Processing: ", fn)
    pcd = o3d.io.read_point_cloud(fn)
    xyz = np.asarray(pcd.points)
    offset = np.array(args.input_offset.split(','), dtype=np.float32).reshape(1, 3)
    if args.factor != 1:
        xyz = pcgc_rescale(xyz + offset, args.factor, args.offset)
    else:
        print('Just adding offset')
        xyz = xyz + offset
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(args.output_ply, pcd, write_ascii=True)
    print("Done: ", args.output_ply)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

