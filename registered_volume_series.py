import regatta.registration_functions as rfunc
import numpy as np
import math
from matplotlib import pyplot as plt
import sys,os,glob

PLOT_SINGLE_VOLUME_RESULTS = False
ALL_CORRS_HISTOGRAM = True

class ReferenceVolume:
    """A class representing the reference volume. This is handy because it
    can store the 3D FFT of the volume such that it doesn't have to be recomputed
    for each B-scan in the target volume."""
    
    def __init__(self,vol):
        self.vol = vol
        self.fref = np.fft.fftn(vol)
        self.n_slow,self.n_depth,self.n_fast = self.vol.shape
        self.rhash = self.get_hash(vol)
        self.cache_folder = './.reference_volume_cache'
        os.makedirs(self.cache_folder,exist_ok=True)
        
    def register(self,target_bscan,poxc=True):
        """Register a single bscan to the reference volume via broadcasting"""
        thash = self.get_hash(target_bscan)
        cachefn = os.path.join(self.cache_folder,'%d_%d.npy'%(self.rhash,thash))
        try:
            xp,yp,zp,xc = np.loadtxt(cachefn)
        except FileNotFoundError:
            ftar = np.conj(np.fft.fft2(target_bscan))
            prod = self.fref*ftar
            if poxc:
                prod = prod/np.abs(prod)
            xc_arr = np.abs(np.fft.ifftn(self.fref*ftar))

            # plt.figure()
            # for k in range(3):
            #     plt.subplot(1,3,k+1)
            #     plt.imshow(np.max(xc_arr,axis=k))
            # plt.show()
            xc = np.max(xc_arr)

            yp,zp,xp = np.unravel_index(np.argmax(xc_arr),xc_arr.shape)
            sy,sz,sx = xc_arr.shape
            #yp = fix(yp,sy)
            zp = self.fix(zp,sz)
            xp = self.fix(xp,sx)
            np.savetxt(cachefn,[xp,yp,zp,xc])
        result = {}
        result['dx'] = xp
        result['dy'] = yp
        result['dz'] = zp
        result['xc'] = xc
        return result

    def fix(self,p,s):
        if p<s//2:
            return p
        else:
            return p-s

    def get_hash(self,vol,N=128):
        return hash(tuple(vol.ravel()[:N]))


class RegisteredVolumeSeries:

    def __init__(self,reference_data):
        self.volume_folders = []
        self.xshifts = []
        self.yshifts = []
        self.zshifts = []
        self.corrs = []
        self.reference_data = reference_data
        self.ref = ReferenceVolume(reference_data)
        #self.add(reference_data)
        
    def add(self,volume_folder,x=None,y=None,z=None,xc=None):

        if x is None or y is None or z is None:
            sy,sz,sx = volume.shape
            x = np.zeros(sy)
            y = np.zeros(sy)
            z = np.zeros(sy)
            xc = np.ones(sy)*np.inf
            
        self.volume_folders.append(volume_folder)
        self.xshifts.append(x)
        self.yshifts.append(y)
        self.zshifts.append(z)
        self.corrs.append(xc)

        

    def register_volumes(self,target_volume_folders_list,show_plots=False,error_rms=0):
        all_corrs = []
        for vidx,target_volume_folder in enumerate(target_volume_folders_list):
            target_volume = rfunc.get_volume(target_volume_folder)
            corr_arr = []
            y_shift_arr = []
            z_shift_arr = []
            x_shift_arr = []
            for s in range(self.ref.n_slow):
                print('Registering B-scan %03d of %03d in volume %03d of %03d.'%(s,self.ref.n_slow,vidx,len(target_volume_folders_list)))
                tar = target_volume[s,:,:]
                res = self.ref.register(tar)

                if error_rms:
                    res['dx'] = res['dx'] + int(np.round(np.random.randn()*error_rms))
                    res['dy'] = res['dy'] + int(np.round(np.random.randn()*error_rms))
                    res['dz'] = res['dz'] + int(np.round(np.random.randn()*error_rms))
                
                corr_arr.append(res['xc'])
                all_corrs.append(res['xc'])
                x_shift_arr.append(res['dx'])
                y_shift_arr.append(res['dy']-s)
                z_shift_arr.append(res['dz'])

            if PLOT_SINGLE_VOLUME_RESULTS:
                plt.figure()
                plt.subplot(2,2,1)
                plt.plot(corr_arr)
                plt.ylabel('xcorr')
                plt.subplot(2,2,2)
                plt.plot(x_shift_arr)
                plt.ylabel('x shift')
                plt.subplot(2,2,3)
                plt.plot(y_shift_arr)
                plt.ylabel('y shift')
                plt.subplot(2,2,4)
                plt.plot(z_shift_arr)
                plt.ylabel('z shift')
                plt.show()
            
            
            # filter shifts using xc here, later

            # ideas for filtering the shift vectors
            # 1. Limit derivative of shift to small amount; motivated
            #    by intuition that the retina's motion between B-scans
            #    is small, esp. at 5 kHz in AO-OCT system
            # 2. Median filtering may remove outliers when they are
            #    isolated, but will fail when clusters of B-scans are out
            #    of place, which appears to happen fequently (during saccades,
            #    for instance)
            # 3. Motion is a non-Markov process; the previous derivative (velocity)
            #    is predictive of the current derivative. This is a stronger statement
            #    than that made in #1 above.
            # 4. Eye movement processes are reversible; that is "previous" in #3 is
            #    equivalent to "next".

            corr_arr = np.array(corr_arr)
            y_shift_arr = np.array(y_shift_arr,dtype=float)
            z_shift_arr = np.array(z_shift_arr,dtype=float)
            x_shift_arr = np.array(x_shift_arr,dtype=float)
            rad_arr = np.sqrt(y_shift_arr**2+z_shift_arr**2+x_shift_arr**2)


            cstd = np.std(corr_arr)
            if cstd>0:
                ncorr_arr = corr_arr/np.std(corr_arr)
            else:
                ncorr_arr = 3.0*np.ones(corr_arr.shape)

            nrad_arr = rad_arr/(np.std(rad_arr)+1)

            ncorr_thresh_std = 1.5
            nrad_thresh_std = 1.0
            valid = (ncorr_arr>ncorr_thresh_std)*(nrad_arr<nrad_thresh_std)
            invalid = 1-valid
            
            corr_arr[np.where(invalid)] = np.nan
            y_shift_arr[np.where(invalid)] = np.nan
            z_shift_arr[np.where(invalid)] = np.nan
            x_shift_arr[np.where(invalid)] = np.nan

            if show_plots:
                plt.cla()
                plt.plot(ncorr_arr,label='normalized correlation')
                plt.plot(nrad_arr,label='normalized 3D displacement')
                for k in range(len(invalid)):
                    if invalid[k]:
                        plt.axvline(k,color='r',alpha=0.2)
                plt.legend()
                os.makedirs('figures',exist_ok=True)
                plt.savefig(os.path.join('figures','regdata_%03d.png'%vidx))
                plt.pause(.1)
            self.add(target_volume_folder,x_shift_arr,y_shift_arr,z_shift_arr,corr_arr)
            
    def correct_volumes(self,output_folder,tags):
        # Determine how large the rendered volume must be.
        # 1. Convert shifts lists to arrays
        # 2. Subtract minimum from each shift array
        # 3. Compute maximum value of shift in each dimension,
        #    and add this to the original volume dimensions

        test = rfunc.get_volume(self.volume_folders[0])
        sy, sz, sx = test.shape

        
        xshifts_vec = np.array(self.xshifts)
        yshifts_vec = np.array(self.yshifts)
        zshifts_vec = np.array(self.zshifts)
        corrs_vec = np.array(self.corrs)

        for arr in [xshifts_vec, yshifts_vec, zshifts_vec, corrs_vec]:
            assert arr.shape[0]==len(tags)
            assert arr.shape[0]==len(self.volume_folders)

        # min-subtract all shifts
        yshifts_vec = yshifts_vec - np.nanmin(yshifts_vec)
        zshifts_vec = zshifts_vec - np.nanmin(zshifts_vec)
        xshifts_vec = xshifts_vec - np.nanmin(xshifts_vec)
        
        # corrected dimensions
        csy = int(sy + np.nanmax(yshifts_vec))
        csz = int(sz + np.nanmax(zshifts_vec))
        csx = int(sx + np.nanmax(xshifts_vec))

        self.corrected_filenames = []
        
        for vfn,ysv,zsv,xsv,cv,tag in zip(self.volume_folders,yshifts_vec,zshifts_vec,xshifts_vec,corrs_vec,tags):
            v = rfunc.get_volume(vfn)
            corrected_volume = np.ones((csy,csz,csx),dtype=complex)*np.nan
            for idx,(bscan,y,z,x,c) in enumerate(zip(v,ysv,zsv,xsv,cv)):
                if np.isnan(c):
                    continue
                corrected_volume[idx+int(y),int(z):int(z+sz),int(x):int(x+sx)] = bscan
                
            # refactor: implement paging for large volume series that can't be held in RAM
            corrected_vfn = os.path.join(output_folder,'%s.npy'%tag)
            self.corrected_filenames.append(corrected_vfn)
            np.save(corrected_vfn,corrected_volume)


    def average_volumes(self,magnitude=True,machine_epsilon=1e-16):
        temp = np.load(self.corrected_filenames[0])

        if magnitude:
            sum_volume = np.zeros(temp.shape,dtype=float)
        else:
            sum_volume = np.zeros(temp.shape,dtype=complex)
            
        counter_volume = np.ones(temp.shape,dtype=float)*machine_epsilon

        for cfn in self.corrected_filenames:
            vol = np.load(cfn)
            vol[np.where(np.isnan(vol))] = 0.0
            if magnitude:
                vol = np.abs(vol)
            sum_volume = sum_volume+vol
            counter_volume[np.where(vol)]+=1

        return sum_volume/counter_volume



    def phase_align_volumes(self,show_plots=False):
        try:
            n_vol = len(self.corrected_volumes)
        except NameError as ne:
            print('This RegisteredVolumeSeries does not have an attribute .corrected_volumes. Make sure correct_volumes has been run.')
            
        sy, sz, sx = self.corrected_volumes[0].shape

        for y in range(sy):
            print('Phase correction %0.1f percent done.'%(100*y/sy))
            for x in range(sx):
                ascans = []
                for v in range(n_vol):
                    # refactor: implement paging for large volume series that can't be held in RAM
                    cv = self.corrected_volumes[v]
                    ascans.append(cv[y,:,x])
                ascans = np.array(ascans,dtype=complex)

                if all(np.isnan(ascans.ravel())):
                    continue
                
                ascans_amplitude = np.abs(ascans)
                ascan_means = np.nanmean(ascans_amplitude,axis=1)
                valid = np.where(1-np.isnan(ascan_means))[0]
                if len(valid)<2:
                    continue
                ref = ascans[valid[0],:]
                corrected_ascans = np.ones(ascans.shape,dtype=complex)*np.nan
                
                for tar_idx in valid:
                    tar = ascans[tar_idx,:]
                    phase_shift = self.get_phase_shift(ref,tar)
                    tar = tar * np.exp(-1j*phase_shift)
                    self.corrected_volumes[tar_idx][y,:,x] = tar
                    corrected_ascans[tar_idx,:] = tar

                if show_plots:
                    plt.figure()
                    plt.subplot(4,1,1)
                    plt.imshow(np.abs(ascans),aspect='auto')
                    plt.subplot(4,1,2)
                    plt.imshow(np.angle(ascans),aspect='auto')
                    plt.subplot(4,1,3)
                    plt.imshow(np.abs(corrected_ascans),aspect='auto')
                    plt.subplot(4,1,4)
                    plt.imshow(np.angle(corrected_ascans),aspect='auto')
                    plt.show()
                

    def get_phase_shift(self,ref,tar,threshold_percentile=80,show_plots=True):
        # simple, non-histogram method for estimating phase shift using
        # the angle of the mean thresholded complex difference between scans
        avg = (np.abs(ref)+np.abs(tar))/2.0
        nonnan_indices = np.where(1-np.isnan(avg))[0]
        avg = avg[nonnan_indices]
        ref = ref[nonnan_indices]
        tar = tar[nonnan_indices]
        
        valid_indices = np.where(avg>=np.percentile(avg,threshold_percentile))
        ref = ref[valid_indices]
        tar = tar[valid_indices]

        phase_shift = np.angle(np.mean(tar*np.conj(ref)))
        return phase_shift
        
                
    def get_phase_shift0(self,ref,tar,threshold_percentile=80,show_plot=True):
        # attempt to use circular, resampled histograms to estimate
        # bulk phase shift; seems to work but generates odd results when
        # an A-scan is corrected to itself; needs debugging
        avg = (np.abs(ref)+np.abs(tar))/2.0
        nonnan_indices = np.where(1-np.isnan(avg))[0]
        avg = avg[nonnan_indices]
        ref = ref[nonnan_indices]
        tar = tar[nonnan_indices]
        
        valid_indices = np.where(avg>=np.percentile(avg,threshold_percentile))
        ref = ref[valid_indices]
        tar = tar[valid_indices]
        bin_lefts = np.arange(0,2*np.pi,np.pi/4.0)
        n_bins = len(bin_lefts)
        
        bin_rights = bin_lefts + np.pi/4.0
        bin_edges = np.array(list(bin_lefts)+[bin_rights[-1]])
        bin_centers = (bin_lefts+bin_rights)/2.0
        
        bin_width = bin_lefts[1]-bin_lefts[0]

        n_shifts = 8

        n_resample = n_bins*n_shifts
        resampled_width = 2*np.pi/n_resample
        
        shifts = np.arange(0,bin_width,bin_width/n_shifts)

        dphase = np.angle(ref)-np.angle(tar)
        if show_plot:
            plt.figure()
            plt.subplot(2,1,1)
            plt.hist(dphase,bins=bin_edges,alpha=0.5,color='tab:blue')
        
        hists = []
        bin_centers = []
        for shift in shifts:
            dphase = (np.angle(ref)-np.angle(tar)-shift)%(2*np.pi)
            hist = np.histogram(dphase,bins=bin_edges)
            hists.append(hist[0])
            bl = hist[1][:-1]
            br = hist[1][1:]
            bc = (bl+br)/2.0+shift
            bin_centers.append(bc)

        hists = np.array(hists).T.ravel()
        bin_centers = np.array(bin_centers).T.ravel()

        if show_plot:
            plt.subplot(2,1,2)
            plt.bar(bin_centers.T.ravel(),hists.T.ravel(),alpha=0.5,color='tab:green',width=resampled_width*0.8)
            plt.show()

        return bin_centers[np.argmax(hists)]

    def compute_correlation(self,v1,v2):
        temp = v1+v2
        non_nan = np.where(1-np.isnan(temp))
        v1valid = v1[non_nan].ravel()
        v2valid = v2[non_nan].ravel()
        corrcoef = np.corrcoef(np.abs(v1valid),np.abs(v2valid))[0,1]
        return corrcoef

    def compute_entropy(self,volume,n_bins=128,bright_layer=False):
        if bright_layer:
            prof = np.nanmean(np.abs(volume),axis=(0,2))
            lidx = np.argmax(prof)
            volume = volume[:,lidx,:]
        non_nan = np.where(1-np.isnan(volume))
        valid = np.abs(volume[non_nan].ravel())
        h,bins = np.histogram(valid,bins=n_bins)
        h = h/len(valid)
        h = h[np.where(np.logical_and(1-np.isnan(h),h>0))]
        entropy = -np.sum(h*np.log2(h))
        return entropy

    def compute_sharpness(self,volume,bright_layer=False):
        if bright_layer:
            prof = np.nanmean(np.abs(volume),axis=(0,2))
            lidx = np.argmax(prof)
            volume = volume[:,lidx,:]
        non_nan = np.where(1-np.isnan(volume))
        valid = np.abs(volume[non_nan].ravel())
        return np.sum(valid**2)/(np.sum(valid)**2)

    def compute_contrast(self,volume,bright_layer=False):
        if bright_layer:
            prof = np.nanmean(np.abs(volume),axis=(0,2))
            lidx = np.argmax(prof)
            volume = volume[:,lidx,:]
        non_nan = np.where(1-np.isnan(volume))
        valid = np.abs(volume[non_nan].ravel())
        M = np.max(valid)
        m = np.min(valid)
        return (M-m)/(M+m)
        


    
