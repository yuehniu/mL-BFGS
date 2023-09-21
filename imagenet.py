"""Prepare the ImageNet dataset"""
import os
import argparse
import tarfile
import pickle
import gzip
import subprocess
from tqdm import tqdm
# from mxnet.gluon.utils import check_sha1
# from gluoncv.utils import download, makedirs

target_dir = './val'
# move images to proper subfolders
val_maps_file = os.path.join( './', 'imagenet_val_maps.pklz')
with gzip.open( val_maps_file, 'rb' ) as f:
    dirs, mappings = pickle.load( f )
for d in dirs:
    os.makedirs( os.path.join(target_dir, d) )
for m in mappings:
    os.rename( os.path.join(target_dir, m[0]), os.path.join(target_dir, m[1], m[0]) )
