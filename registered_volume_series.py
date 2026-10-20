import regatta.registration_functions as rfunc
import numpy as np
import math
from matplotlib import pyplot as plt
import sys,os,glob
from pathlib import Path
import scipy.ndimage as spn
import scipy.interpolate as spi
PLOT_SINGLE_VOLUME_RESULTS = False
ALL_CORRS_HISTOGRAM = True

class RegisteredVolumeSeries:
    def __init__(self,manifest):
        self.reference_volume_filename = manifest.reference_volume_filename
        self.target_volume_filenames = manifest.target_volume_filenames
        self.reference_tag = self.make_tag(self.reference_volume_filename)
        self.tag_dictionary = dict(zip(self.target_volume_filenames,[self.make_tag(f) for f in self.target_volume_filenames]))
        self.reference_volume = rfunc.get_volume(self.reference_volume_filename)
        self.n_slow,self.n_depth,self.n_fast = self.reference_volume.shape

        self.output_folder = manifest.output_folder
        self.cache_folder = os.path.join(self.output_folder,'.registration_cache')
        self.figure_folder = os.path.join(self.output_folder,'figures')
        self.registration_folder = os.path.join(self.output_folder,'registration')
        
        os.makedirs(self.output_folder,exist_ok=True)
        os.makedirs(self.cache_folder,exist_ok=True)
        os.makedirs(self.figure_folder,exist_ok=True)
        os.makedirs(self.registration_folder,exist_ok=True)

        try:
            self.fref = np.load(os.path.join(self.cache_folder,'fref.npy'))
        except FileNotFoundError:
            self.fref = np.fft.fftn(self.reference_volume)
            np.save(os.path.join(self.cache_folder,'fref.npy'),self.fref)

        self.reference_energy = np.sum(np.abs(self.reference_volume)**2)/self.n_slow
        self.rhash = self.get_hash(self.reference_volume)

    def get_hash(self,vol,N=128):
        return hash(tuple(vol.ravel()[:N]))

    def fix(self,p,s):
        if p<s//2:
            return p
        else:
            return p-s

    def make_tag(self,filename,n_terms=3):
        path = Path(filename)
        path_list = list(path.parts)
        return '_'.join(path_list[-n_terms:])
        
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

            # get the non-normalized cross-correlation:
            xc = np.max(xc_arr)
            # compute target energy for cross-correlation normalization:
            target_energy = np.sum(np.abs(target_bscan)**2)
            xc = xc/np.sqrt(self.reference_energy*target_energy)

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

    def add(self,volume_filename,x,y,z,xc):

        self.volume_filenames.append(volume_filename)
        self.xshifts.append(x)
        self.yshifts.append(y)
        self.zshifts.append(z)
        self.corrs.append(xc)

    def get_registration_filenames(self,target_volume_filename):
        registration_tag = 'ref_%s_tar_%s'%(self.reference_tag,
                                            self.tag_dictionary[target_volume_filename])
        corr_fn = os.path.join(self.registration_folder,'%s_corr.npy'%registration_tag)
        dx_fn = os.path.join(self.registration_folder,'%s_dx.npy'%registration_tag)
        dy_fn = os.path.join(self.registration_folder,'%s_dy.npy'%registration_tag)
        dz_fn = os.path.join(self.registration_folder,'%s_dz.npy'%registration_tag)
        return corr_fn,dx_fn,dy_fn,dz_fn
        
    def compute_cross_correlations(self,show_plots=False):
        target_volume_filenames = self.target_volume_filenames
        all_corrs = []
        all_dxs = []
        all_dys = []
        all_dzs = []
        for vidx,target_volume_filename in enumerate(target_volume_filenames):
            target_volume = rfunc.get_volume(target_volume_filename)
            corr_fn,dx_fn,dy_fn,dz_fn = self.get_registration_filenames(target_volume_filename)
            try:
                corr_arr = np.load(corr_fn)
                dx_arr = np.load(dx_fn)
                dy_arr = np.load(dy_fn)
                dz_arr = np.load(dz_fn)
            except FileNotFoundError:
                corr_arr = []
                dy_arr = []
                dz_arr = []
                dx_arr = []
                for s in range(self.n_slow):
                    print('Registering B-scan %03d of %03d in volume %03d of %03d.'%(s,self.n_slow,vidx,len(target_volume_filenames)))
                    tar = target_volume[s,:,:]
                    res = self.register(tar)

                    corr_arr.append(res['xc'])
                    dx_arr.append(res['dx'])
                    dy_arr.append(res['dy']-s)
                    dz_arr.append(res['dz'])

                corr_arr = np.array(corr_arr)
                dx_arr = np.array(dx_arr)
                dy_arr = np.array(dy_arr)
                dz_arr = np.array(dz_arr)
                np.save(corr_fn,corr_arr)
                np.save(dx_fn,dx_arr)
                np.save(dy_fn,dy_arr)
                np.save(dz_fn,dz_arr)
                
            all_corrs.append(corr_arr)
            all_dxs.append(dx_arr)
            all_dys.append(dy_arr)
            all_dzs.append(dz_arr)

        all_corrs = np.array(all_corrs)
        all_dxs = np.array(all_dxs)
        all_dys = np.array(all_dys)
        all_dzs = np.array(all_dzs)
        result = {}
        result['xc'] = all_corrs
        result['dx'] = all_dxs
        result['dy'] = all_dys
        result['dz'] = all_dzs
        if show_plots:
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(all_corrs,aspect='auto',interpolation='none')
            plt.title('correlations')
            plt.ylabel('B-scan index')
            plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow(all_dxs,aspect='auto',interpolation='none')
            plt.title('dx')
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.imshow(all_dys,aspect='auto',interpolation='none')
            plt.title('dy')
            plt.ylabel('B-scan index')
            plt.xlabel('A-scan index')
            plt.colorbar()
            plt.subplot(2,2,4)
            plt.imshow(all_dzs,aspect='auto',interpolation='none')
            plt.title('dz')
            plt.xlabel('A-scan index')
            plt.colorbar()
            plt.suptitle('cross correlation results')
            plt.savefig(os.path.join(self.figure_folder,'compute_cross_correlations.png'))
            plt.show()
        return result

        
    def compute_trajectories(self,show_plots=False,specific_index=None):
        
        target_volume_filenames = self.target_volume_filenames
        for vidx,target_volume_filename in enumerate(target_volume_filenames):
            if specific_index is not None:
                if not vidx==specific_index:
                    continue

            corr_fn,dx_fn,dy_fn,dz_fn = self.get_registration_filenames(target_volume_filename)
            corr_arr = np.load(corr_fn)
            dx_arr = np.load(dx_fn).astype(float)
            dy_arr = np.load(dy_fn).astype(float)
            dz_arr = np.load(dz_fn).astype(float)
            
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


            t = np.arange(len(corr_arr))
            
            # compute Euclidean displacement array from x, y, z
            rad_arr = np.sqrt(dy_arr**2+dz_arr**2+dx_arr**2)

            # compute a median-filtered version of rad_arr
            rad_medfilt = spn.median_filter(rad_arr,size=13,mode='nearest')

            # subtract the median-filtered version from the original rad_arr
            medfilt_subtracted = np.abs(rad_arr-rad_medfilt)


            plt.figure()
            plt.subplot(1,3,1)
            plt.plot(t,rad_arr)
            plt.title('displacement')
            plt.subplot(1,3,2)
            plt.plot(t,rad_medfilt)
            plt.title('displacement median filtered')
            plt.subplot(1,3,3)
            plt.plot(t,medfilt_subtracted)
            plt.title('displacement corrected')

            threshold = 5
            valid = np.where(medfilt_subtracted<=threshold)
            invalid = np.where(medfilt_subtracted>threshold)
            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(t[valid],dx_arr[valid],'go')
            plt.plot(t[invalid],dx_arr[invalid],'ro')
            plt.title('dx')
            plt.subplot(2,2,2)
            plt.plot(t[valid],dy_arr[valid],'go')
            plt.plot(t[invalid],dy_arr[invalid],'ro')
            plt.title('dy')
            plt.subplot(2,2,3)
            plt.plot(t[valid],dz_arr[valid],'go')
            plt.plot(t[invalid],dz_arr[invalid],'ro')
            plt.title('dz')
            plt.subplot(2,2,4)
            plt.plot(t[valid],medfilt_subtracted[valid],'go')
            plt.plot(t[invalid],medfilt_subtracted[invalid],'ro')
            plt.title('displacement, corrected')
            
            # 2. Fit a cubic B-Spline (k=3) for each axis relative to time 't'
            spline_k = 3
            spline_x = spi.make_interp_spline(t[valid], dx_arr[valid], k=spline_k)
            spline_y = spi.make_interp_spline(t[valid], dy_arr[valid], k=spline_k)
            spline_z = spi.make_interp_spline(t[valid], dz_arr[valid], k=spline_k)

            # 3. Evaluate the smooth path on a denser time grid
            x_smooth = spline_x(t)
            y_smooth = spline_y(t)
            z_smooth = spline_z(t)

            # 4. Plot the results in 3D
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(dx_arr[invalid], dy_arr[invalid], dz_arr[invalid], color='red', label='invalid')
            ax.scatter(dx_arr[valid], dy_arr[valid], dz_arr[valid], color='green', label='valid')
            ax.plot(x_smooth, y_smooth, z_smooth, color='blue', label='fitted spline')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            ax.legend()

            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(t[invalid],dx_arr[invalid],'ro',alpha=0.25)
            plt.plot(t[valid],dx_arr[valid],'go',alpha=0.25)
            plt.plot(t,x_smooth,'k-')
            plt.subplot(2,2,2)
            plt.plot(t[invalid],dy_arr[invalid],'ro',alpha=0.25)
            plt.plot(t[valid],dy_arr[valid],'go',alpha=0.25)
            plt.plot(t,y_smooth,'k-')
            plt.subplot(2,2,3)
            plt.plot(t[invalid],dz_arr[invalid],'ro',alpha=0.25)
            plt.plot(t[valid],dz_arr[valid],'go',alpha=0.25)
            plt.plot(t,z_smooth,'k-')
            plt.subplot(2,2,4)
            plt.plot(t[valid],medfilt_subtracted[valid],'go')
            plt.plot(t[invalid],medfilt_subtracted[invalid],'ro')

            
            plt.show()
            
    def correct_volumes(self):
        # Determine how large the rendered volume must be.
        # 1. Convert shifts lists to arrays
        # 2. Subtract minimum from each shift array
        # 3. Compute maximum value of shift in each dimension,
        #    and add this to the original volume dimensions

        # refactor: implement paging for large volume series that can't be held in RAM

        xshifts = []
        yshifts = []
        zshifts = []
        corrs = []
        
        # First loop: iterate through and assemble all the registration data:
        for target_index,target_volume_filename in enumerate(self.target_volume_filenames):
            corr_fn,dx_fn,dy_fn,dz_fn = self.get_registration_filenames(target_volume_filename)
            corrs.append(np.load(corr_fn))
            xshifts.append(np.load(dx_fn).astype(float))
            yshifts.append(np.load(dy_fn).astype(float))
            zshifts.append(np.load(dz_fn).astype(float))
            
        
        xshifts_vec = np.array(xshifts)
        yshifts_vec = np.array(yshifts)
        zshifts_vec = np.array(zshifts)
        corrs_vec = np.array(corrs)
        
        # min-subtract all shifts
        yshifts_vec = yshifts_vec - np.nanmin(yshifts_vec)
        zshifts_vec = zshifts_vec - np.nanmin(zshifts_vec)
        xshifts_vec = xshifts_vec - np.nanmin(xshifts_vec)
        
        # corrected dimensions
        csy = int(self.n_slow + np.nanmax(yshifts_vec))
        csz = int(self.n_depth + np.nanmax(zshifts_vec))
        csx = int(self.n_fast + np.nanmax(xshifts_vec))

        for target_volume_filename,ysv,zsv,xsv,cv in zip(self.target_volume_filenames,yshifts_vec,zshifts_vec,xshifts_vec,corrs_vec):
            output_filename = os.path.join(self.output_folder,'corrected_%s.npy'%self.tag_dictionary[target_volume_filename])
            if os.path.exists(output_filename):
                print('Corrected volume %s exists, skipping.'%output_filename)
                continue

            
            target_volume = rfunc.get_volume(target_volume_filename)
            corrected_volume = np.ones((csy,csz,csx),dtype=complex)*np.nan

            for idx,(bscan,y,z,x,c) in enumerate(zip(target_volume,ysv,zsv,xsv,cv)):
                if np.isnan(c):
                    continue
                corrected_volume[idx+int(y),int(z):int(z+self.n_depth),int(x):int(x+self.n_fast)] = bscan

            print('Saving %s.'%output_filename)
            np.save(output_filename,corrected_volume)

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
        


    
