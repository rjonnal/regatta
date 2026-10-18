"""
Created on 29 January 2026
@author:RSJ

In this test we test the `register` method of `reference_image.ReferenceImage`,
using real AO-OCT small volumes.
"""
import regatta
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob

from regatta import load_image_aooct as load

reference_index = 0
target_index = 0
force_random = True

data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct_1deg/'
#data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct/'
#data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_synthetic/'

volume_locations = sorted(glob.glob(os.path.join(data_root,'*')))
reference_location = volume_locations[reference_index]

ref_data = load(reference_location)
tar_data = load(volume_locations[target_index])


def flythrough(data):
    a_data = 20*np.log10(np.abs(data))

    sy,sz,sx = a_data.shape

    cmin,cmax = np.percentile(a_data,(30,99.9))
    
    for z in range(sz):
        plt.cla()
        plt.imshow(a_data[:,z,:],clim=(cmin,cmax),cmap='gray')
        plt.title('%0.1f dB - %0.1f dB'%(cmin,cmax))
        plt.pause(.1)


def project_3(data,projection_func=np.max,block_swap=False):
    a_data = np.abs(data)
    if block_swap:
        a_data = np.fft.fftshift(a_data)
    plt.figure()
    for k in range(3):
        plt.subplot(1,3,k+1)
        plt.imshow(projection_func(a_data,axis=k),cmap='gray')
        


fref = np.conj(np.fft.fftn(ref_data))

sy,sz,sx = tar_data.shape
pyvec = []
pxvec = []
pzvec = []
corrvec = []

YY,ZZ,XX = np.meshgrid(np.arange(sy)-sy/2.0,np.arange(sz)-sz/2.0,np.arange(sx)-sx/2.0,indexing='ij')

sigma = 100
gfilter = np.exp(-(YY**2+ZZ**2+XX**2)/(2*sigma**2))
gfilter = np.fft.fftshift(gfilter)
gfilter = gfilter/np.mean(gfilter)

def get_registration_info(nxc):
    sy,sz,sx = nxc.shape
    py,pz,px = np.unravel_index(np.argmax(nxc),nxc.shape)
    #py = py if py < sy //2 else py - sy
    pz = pz if pz < sz //2 else pz - sz
    px = px if px < sx //2 else px - sx
    return py,pz,px,np.max(nxc)*sy

for y in range(0,sy,5):
    print('%d of %d'%(y,sy))
    tar = tar_data[y,:,:]
    ftar = np.fft.fftn(tar)
    prod = ftar*fref
    prod = prod/(np.abs(prod)+1e-20)
    nxc = np.fft.ifftn(prod)
    nxc = np.abs(nxc)
    normfunc = np.std
    
    nxc = nxc/normfunc(nxc,axis=0)*normfunc(nxc)
    nxc = np.transpose(nxc,(1,2,0))
    nxc = nxc/normfunc(nxc,axis=0)*normfunc(nxc)
    nxc = np.transpose(nxc,(1,2,0))
    nxc = nxc/normfunc(nxc,axis=0)*normfunc(nxc)
    nxc = np.transpose(nxc,(1,2,0))
    #project_3(nxc,block_swap=True)
    #plt.show()
    py,pz,px,corr = get_registration_info(nxc)
    print(py,pz,px,corr)
    pyvec.append(py)
    pzvec.append(pz)
    pxvec.append(px)
    corrvec.append(corr)
    
plt.figure()
plt.plot(pyvec)
plt.plot(pzvec)
plt.plot(pxvec)
plt.figure()
plt.plot(corrvec)
plt.show()








        
# class RegisteredImageSeries:

#     def __init__(self,reference_data,oversampling=1):
#         self.images = []
#         self.images.append((reference_data,(0,0,0,1.0)))
#         self.fref = np.conj(np.fft.fftn(reference_data))
#         self.machine_epsilon = np.finfo(float).eps
#         self.oversampling = oversampling
        
#     def add_whole(self,target_data,poxc=True):
#         sy,sz,sx = target_data.shape
        
#         ftar = np.fft.fftn(target_data)
#         prod = ftar*self.fref
#         if poxc:
#             prod = prod / (np.abs(prod) + self.machine_epsilon)

#         if self.oversampling==1:
#             nxc = np.real(np.fft.ifftn(prod))
#         else:
#             nxc = np.real(np.fft.ifftn(prod,s=(sy*self.oversampling,
#                                                sz*self.oversampling,
#                                                sx*self.oversampling)))

#         nxc = nxc/np.std(nxc,axis=0)*np.std(nxc)
#         nxc = np.transpose(nxc,(1,2,0))
#         nxc = nxc/np.std(nxc,axis=0)*np.std(nxc)
#         nxc = np.transpose(nxc,(1,2,0))
#         nxc = nxc/np.std(nxc,axis=0)*np.std(nxc)
#         nxc = np.transpose(nxc,(1,2,0))
        
#         py,pz,px = np.unravel_index(np.argmax(nxc),nxc.shape)
#         py = self.wrap_fix(py,sy*self.oversampling)
#         pz = self.wrap_fix(pz,sz*self.oversampling)
#         px = self.wrap_fix(px,sx*self.oversampling)
        
#         correlation = nxc[py,pz,px]
#         registration_info = (py,pz,px,correlation)
#         self.images.append((target_data,registration_info))
#         print(registration_info)
#         project_3(nxc,block_swap=True)
#         project_3(target_data)
#         project_3(self.images[0][0])
#         plt.show()

#     def add_strips(self,target_data,strip_width,poxc=True):
#         sy,sz,sx = target_data.shape
        
#         ftar = np.fft.fftn(target_data)
#         prod = ftar*self.fref
#         if poxc:
#             prod = prod / (np.abs(prod) + self.machine_epsilon)

#         if self.oversampling==1:
#             nxc = np.real(np.fft.ifftn(prod))
#         else:
#             nxc = np.real(np.fft.ifftn(prod,s=(sy*self.oversampling,
#                                                sz*self.oversampling,
#                                                sx*self.oversampling)))

#         nxc = nxc/np.std(nxc,axis=0)*np.std(nxc)
#         nxc = np.transpose(nxc,(1,2,0))
#         nxc = nxc/np.std(nxc,axis=0)*np.std(nxc)
#         nxc = np.transpose(nxc,(1,2,0))
#         nxc = nxc/np.std(nxc,axis=0)*np.std(nxc)
#         nxc = np.transpose(nxc,(1,2,0))
        
#         py,pz,px = np.unravel_index(np.argmax(nxc),nxc.shape)
#         py = self.wrap_fix(py,sy*self.oversampling)
#         pz = self.wrap_fix(pz,sz*self.oversampling)
#         px = self.wrap_fix(px,sx*self.oversampling)
        
#         correlation = nxc[py,pz,px]
#         registration_info = (py,pz,px,correlation)
#         self.images.append((target_data,registration_info))
#         print(registration_info)
#         project_3(nxc,block_swap=True)
#         project_3(target_data)
#         project_3(self.images[0][0])
#         plt.show()

        
#     def wrap_fix(self, p, size):
#         # identical wrap convention used in testing_broadcasting.py (fix on z,x; not y)
#         return p if p < size // 2 else p - size
        
# ris = RegisteredImageSeries(ref_data,oversampling=1)

# #for tidx in range(10):
# #    ris.add(load(volume_locations[tidx]),poxc=True)

# #flythrough(ref_data)


# def get_registration_info(nxc):
#     sy,sz,sx = nxc.shape
#     py,pz,px = np.unravel_index(np.argmax(nxc),nxc.shape)
#     py = py if py < sy //2 else py - sy
#     pz = pz if pz < sz //2 else pz - sz
#     px = px if px < sx //2 else px - sx
#     return py,pz,px,np.max(nxc)

# strip_width = 20
# sy,sz,sx = tar_data.shape

# strip_starts = list(range(0,sy,strip_width))
# strip_ends = strip_starts[1:]+[sy]


# fref = np.conj(np.fft.fftn(ref_data))

# pyvec = []
# pxvec = []
# pzvec = []

# for s,e in zip(strip_starts,strip_ends):
#     tar = np.zeros((sy,sz,sx),dtype=complex)
#     tar[s:e,:,:] = tar_data[s:e,:,:]
#     ftar = np.fft.fftn(tar)
#     prod = ftar*fref
#     #prod = prod/np.abs(prod+1e-20)
#     nxc = np.abs(np.fft.ifftn(prod))
#     py,pz,px,correlation = get_registration_info(nxc)
#     pyvec.append(py)
#     pzvec.append(pz)
#     pxvec.append(px)
    
# plt.plot(pyvec)
# plt.plot(pzvec)
# plt.plot(pxvec)
# plt.show()
