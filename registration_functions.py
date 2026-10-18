import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
import scipy.interpolate as spi
import scipy.signal as sps
#import inspect
#from matplotlib.widgets import Button, Slider, SpanSelector, Cursor
import pandas as pd
import json

def load_dict(fn):
    with open(fn,'r') as fid:
        s = fid.read()
        d = json.loads(s)
    return d

def save_dict(fn,d):
    s = json.dumps(d)
    with open(fn,'w') as fid:
        fid.write(s)

def shear(bscan,max_roll):
    out = np.zeros(bscan.shape,dtype=complex)
    roll_vec = np.linspace(0,max_roll,bscan.shape[1])
    roll_vec = np.round(roll_vec).astype(int)
    for k in range(bscan.shape[1]):
        out[:,k] = np.roll(bscan[:,k],roll_vec[k])
    return out

def get_xflattening_function(bscan,min_shift=-30,max_shift=30,diagnostics=False,do_plot=False):
    shift_range = range(min_shift,max_shift) # [-20, -19, ..... 19, 20]
    peaks = np.zeros(len(shift_range)) # [0, 0, ..... 0, 0]
    profs = []
    for idx,shift in enumerate(shift_range): # iterate through [-20, -19, ..... 19, 20]
        temp = shear(bscan,shift) # shear by -20, then -19, then -18...
        prof = np.mean(np.abs(temp),axis=1) # compute the lateral median
        profs.append(prof)
        peaks[idx] = np.max(prof) # replace the 0 in peaks with whatever the max value is of prof
    # now, find the location of the highest value in peaks, and use that index to find the optimal shift
    optimal_shift = shift_range[np.argmax(peaks)]
    profs = np.array(profs)
    if do_plot:
        fig = plt.figure(figsize=(6,8))
        ax = fig.subplots(2,1)
        ax[0].imshow(profs.T,aspect='auto')
        ax[1].plot(shift_range,peaks)
        ax[1].set_xlabel('max shear')
        ax[1].set_ylabel('max profile peak')
        plt.show()
    return lambda bscan: shear(bscan,optimal_shift)


def flatten_volume(volume):
    n_slow,n_fast,n_depth = volume.shape
    temp = np.mean(np.abs(volume[n_slow//2-3:n_slow//2+3,:,:]),axis=0)
    f = get_xflattening_function(temp)
    flattened = []
    for s in range(n_slow):
        flattened.append(f(volume[s,:,:]))

    flattened = np.array(flattened,dtype=complex)
    targets = np.mean(np.abs(flattened),axis=2)
    plt.figure()
    plt.imshow(targets)
    
    reference = np.mean(targets[n_slow//2-3:n_slow//2+3,:],axis=0)
    shifts = [xcorr(t,reference) for t in targets]
    shifts = sps.medfilt(shifts,5)
    out = []
    out = [np.roll(b,s,axis=0) for b,s in zip(flattened,shifts)]
    out = np.array(out)
    return out

def generate_registration_manifest(folder, reference_label, upsample_factor = 2):
    outfn = folder + '_registration_manifest.json'
    flist = glob.glob(os.path.join(folder,'*'))
    flist.sort()
    d = {}
    d['reference'] = reference_label
    d['targets'] = flist
    d['upsample_factor'] = upsample_factor
    save_dict(outfn,d)

def get_volume(folder,prefix=''):
    flist = glob.glob(os.path.join(folder,'%s*.npy'%prefix))
    flist.sort()
    
    vol = [np.load(f) for f in flist]
    vol = np.array(vol)
    return vol#[:10,:20,:30]

def upsample(vol,factor):
    return vol.repeat(factor,axis=0).repeat(factor,axis=1).repeat(factor,axis=2)


def nxc3(a,b):
    prod = np.fft.fftn(a)*np.conj(np.fft.fftn(b))
    aprod = np.abs(prod)+1e-16
    #prod = prod/aprod
    out = np.abs(np.fft.ifftn(prod))
    return out

def flythrough3(temp,fps=5,dB=True):
    if dB:
        a = 20*np.log10(temp)
    else:
        a = temp
        
    amean = np.nanmean(a)
    for k in range(len(a.shape)):
        for d in range(a.shape[0]):
            frame = a[d,:,:]
            if np.nanmean(frame)<amean/2.0:
                continue
            if np.all(np.isnan(frame)):
                continue
            plt.cla()
            plt.imshow(a[d,:,:],cmap='gray')
            plt.title('dim %d frame %d'%(k,d))
            plt.pause(1.0/fps)
        a = np.transpose(a,[1,2,0])
        

def project3(a,pfunc=np.nanmean):
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.imshow(pfunc(a,axis=0),cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(pfunc(a,axis=1),cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(pfunc(a,axis=2),cmap='gray')
    plt.show()

def project3multiple(tup,pfunc=np.nanmean,clim=(None,None)):

    nim = len(tup)
    
    plt.figure(figsize=(9,3*nim))
    for row in range(nim):
        for col in range(3):
            plt.subplot(nim,3,row*3+col+1)
            im = pfunc(tup[row],axis=col)
            im[np.where(np.isnan(im))] = np.nanmin(im)
            im[np.where(im==0)] = np.nanmin(im)
            if col==2:
                plt.imshow(im.T,cmap='gray',clim=clim)
            else:
                plt.imshow(im,cmap='gray',clim=clim)
                
    
    plt.show()


def brightest(vol,axis=0,half_width=1,dB=True):
    sy,sz,sx = vol.shape
    py = np.argmax(np.nanmean(np.abs(vol),axis=(1,2)))
    pz = np.argmax(np.nanmean(np.abs(vol),axis=(0,2)))
    px = np.argmax(np.nanmean(np.abs(vol),axis=(0,1)))

    if axis==0:
        out = np.mean(np.abs(vol[py-half_width:py+half_width+1,:,:]),axis=axis)
    if axis==1:
        out = np.mean(np.abs(vol[:,pz-half_width:pz+half_width+1,:]),axis=axis)
    if axis==2:
        out = np.mean(np.abs(vol[:,:,px-half_width:px+half_width+1]),axis=axis)

    if dB:
        out = 20*np.log10(out)
    return out



def reconcile_volume_sizes(data_root,corrected_suffix='_corrected'):
    corrected_root = data_root + corrected_suffix

    folder_list = glob.glob(os.path.join(data_root,'*'))
    folder_list.sort()

    symax = -1
    sxmax = -1
    szmax = -1

    for f in folder_list:
        bscan_list = glob.glob(os.path.join(f,'*.npy'))
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
        bscan_list = glob.glob(os.path.join(f,'*.npy'))
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
            print('Writing %s.'%outfn)
