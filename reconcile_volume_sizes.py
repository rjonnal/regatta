import numpy as np
from matplotlib import pyplot as plt
import glob
import os,sys

data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct_1deg'
corrected_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct_1deg_corrected'

folder_list = glob.glob(os.path.join(data_root,'*'))
folder_list.sort()

symax = -1
sxmax = -1
szmax = -1

for f in folder_list:
    bscan_list = glob.glob(os.path.join(f,'*bscan*.npy'))
    bscan_list.sort()
    if len(bscan_list)>symax:
        symax=len(bscan_list)
    for b in bscan_list:
        bscan = np.load(b)
        sz,sx = bscan.shape
        if sz>szmax:
            szmax=sz
        if sx>sxmax:
            sxmax=sx

for f in folder_list:
    bscan_list = glob.glob(os.path.join(f,'*bscan*.npy'))
    bscan_list.sort()
    old_volume = np.array([np.load(b) for b in bscan_list])
    new_volume = np.zeros((symax,szmax,sxmax),dtype=complex)
    sy,sz,sx = old_volume.shape
    new_volume[:sy,:sz,:sx] = old_volume
    
    new_f = os.path.join(corrected_root,os.path.split(f)[1])
    os.makedirs(new_f,exist_ok=True)
    for y in range(symax):
        outfn = os.path.join(new_f,'bscan_%05d.npy'%y)
        np.save(outfn,new_volume[y,:,:])

