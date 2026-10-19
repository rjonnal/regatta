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

class ReferenceVolume:
    """A class representing the reference volume. This is handy because it
    can store the 3D FFT of the volume such that it doesn't have to be recomputed
    for each B-scan in the target volume."""
    
    def __init__(self,vol,output_folder='.'):
        self.vol = vol
        self.output_folder = output_folder
        self.fref = np.fft.fftn(vol)
        self.n_slow,self.n_depth,self.n_fast = self.vol.shape
        self.rhash = self.get_hash(vol)
        self.cache_folder = os.path.join(self.output_folder,'.reference_volume_cache')
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

    def __init__(self,reference_data,manifest):
        self.volume_filenames = []
        self.corrected_filename_dictionary = {}
        self.xshifts = []
        self.yshifts = []
        self.zshifts = []
        self.corrs = []
        self.manifest = manifest
        self.target_volume_filenames = manifest.target_volume_filenames
        self.tag_dictionary = dict(zip(self.target_volume_filenames,[self.make_tag(f) for f in self.target_volume_filenames]))
        self.output_folder = self.manifest.output_folder
        self.ref = ReferenceVolume(reference_data,self.output_folder)

    def make_tag(self,filename,n_terms=2):
        path = Path(filename)
        path_list = list(path.parts)
        return '_'.join(path_list[-n_terms:])
        
    def add(self,volume_filename,x,y,z,xc):

        self.volume_filenames.append(volume_filename)
        self.xshifts.append(x)
        self.yshifts.append(y)
        self.zshifts.append(z)
        self.corrs.append(xc)

    def register_volumes_simple(self,show_plots=False):
        target_volume_filenames = self.target_volume_filenames
        all_corrs = []
        for vidx,target_volume_filename in enumerate(target_volume_filenames):
            target_volume = rfunc.get_volume(target_volume_filename)
            corr_arr = []
            y_shift_arr = []
            z_shift_arr = []
            x_shift_arr = []
            for s in range(self.ref.n_slow):
                print('Registering B-scan %03d of %03d in volume %03d of %03d.'%(s,self.ref.n_slow,vidx,len(target_volume_filenames)))
                tar = target_volume[s,:,:]
                res = self.ref.register(tar)
                corr_arr.append(res['xc'])
                all_corrs.append(res['xc'])
                x_shift_arr.append(res['dx'])
                y_shift_arr.append(res['dy']-s)
                z_shift_arr.append(res['dz'])

            self.add(target_volume_filename,
                     x_shift_arr_valid,
                     y_shift_arr_valid,
                     z_shift_arr_valid,
                     corr_arr_valid)

    def register_volumes(self,show_plots=False):
        target_volume_filenames = self.target_volume_filenames
        all_corrs = []
        for vidx,target_volume_filename in enumerate(target_volume_filenames):
            target_volume = rfunc.get_volume(target_volume_filename)
            corr_arr = []
            y_shift_arr = []
            z_shift_arr = []
            x_shift_arr = []
            for s in range(self.ref.n_slow):
                print('Registering B-scan %03d of %03d in volume %03d of %03d.'%(s,self.ref.n_slow,vidx,len(target_volume_filenames)))
                tar = target_volume[s,:,:]
                res = self.ref.register(tar)

                corr_arr.append(res['xc'])
                all_corrs.append(res['xc'])
                x_shift_arr.append(res['dx'])
                y_shift_arr.append(res['dy']-s)
                z_shift_arr.append(res['dz'])

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

            # convert our accummulation lists into arrays:
            corr_arr = np.array(corr_arr)
            y_shift_arr = np.array(y_shift_arr,dtype=float)
            z_shift_arr = np.array(z_shift_arr,dtype=float)
            x_shift_arr = np.array(x_shift_arr,dtype=float)

            t = np.arange(len(corr_arr))
            
            
            # compute Euclidean displacement array from x, y, z
            rad_arr = np.sqrt(y_shift_arr**2+z_shift_arr**2+x_shift_arr**2)

            # compute a median-filtered version of rad_arr
            rad_medfilt = spn.median_filter(rad_arr,size=13,mode='nearest')

            # subtract the median-filtered version from the original rad_arr
            medfilt_subtracted = np.abs(rad_arr-rad_medfilt)


            plt.figure()
            plt.subplot(1,3,1)
            plt.plot(t,rad_arr)
            plt.subplot(1,3,2)
            plt.plot(t,rad_medfilt)
            plt.subplot(1,3,3)
            plt.plot(t,medfilt_subtracted)

            threshold = 5
            valid = np.where(medfilt_subtracted<=threshold)
            invalid = np.where(medfilt_subtracted>threshold)
            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(t[valid],x_shift_arr[valid],'go')
            plt.plot(t[invalid],x_shift_arr[invalid],'ro')
            plt.subplot(2,2,2)
            plt.plot(t[valid],y_shift_arr[valid],'go')
            plt.plot(t[invalid],y_shift_arr[invalid],'ro')
            plt.subplot(2,2,3)
            plt.plot(t[valid],z_shift_arr[valid],'go')
            plt.plot(t[invalid],z_shift_arr[invalid],'ro')
            plt.subplot(2,2,4)
            plt.plot(t[valid],medfilt_subtracted[valid],'go')
            plt.plot(t[invalid],medfilt_subtracted[invalid],'ro')
            plt.show()
            
            # 2. Fit a cubic B-Spline (k=3) for each axis relative to time 't'
            spline_k = 1
            spline_x = spi.make_interp_spline(t[valid], x_shift_arr[valid], k=spline_k)
            spline_y = spi.make_interp_spline(t[valid], y_shift_arr[valid], k=spline_k)
            spline_z = spi.make_interp_spline(t[valid], z_shift_arr[valid], k=spline_k)

            # 3. Evaluate the smooth path on a denser time grid
            x_smooth = spline_x(t)
            y_smooth = spline_y(t)
            z_smooth = spline_z(t)

            # 4. Plot the results in 3D
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x_shift_arr[invalid], y_shift_arr[invalid], z_shift_arr[invalid], color='red', label='invalid')
            ax.scatter(x_shift_arr[valid], y_shift_arr[valid], z_shift_arr[valid], color='green', label='valid')
            ax.plot(x_smooth, y_smooth, z_smooth, color='blue', label='fitted spline')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            ax.legend()
            plt.show()

            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(t[invalid],x_shift_arr[invalid],'ro',alpha=0.25)
            plt.plot(t[valid],x_shift_arr[valid],'go',alpha=0.25)
            plt.plot(t,x_smooth,'k-')
            plt.subplot(2,2,2)
            plt.plot(t[invalid],y_shift_arr[invalid],'ro',alpha=0.25)
            plt.plot(t[valid],y_shift_arr[valid],'go',alpha=0.25)
            plt.plot(t,y_smooth,'k-')
            plt.subplot(2,2,3)
            plt.plot(t[invalid],z_shift_arr[invalid],'ro',alpha=0.25)
            plt.plot(t[valid],z_shift_arr[valid],'go',alpha=0.25)
            plt.plot(t,z_smooth,'k-')
            plt.subplot(2,2,4)
            plt.plot(t[valid],medfilt_subtracted[valid],'go')
            plt.plot(t[invalid],medfilt_subtracted[invalid],'ro')

            
            plt.show()
            sys.exit()

            # # 1. Generate some dummy noisy 3D trajectory data
            # np.random.seed(42)
            # t = np.linspace(0, 10, num=20)  # Time steps
            # x = 2 * np.sin(t) + np.random.normal(0, 0.1, size=20)
            # y = 2 * np.cos(t) + np.random.normal(0, 0.1, size=20)
            # z = t * 0.5 + np.random.normal(0, 0.1, size=20)
            # points = np.column_stack((x, y, z))

            # # 2. Fit a cubic B-Spline (k=3) for each axis relative to time 't'
            # spline_x = spi.make_interp_spline(t, x, k=3)
            # spline_y = spi.make_interp_spline(t, y, k=3)
            # spline_z = spi.make_interp_spline(t, z, k=3)

            # # 3. Evaluate the smooth path on a denser time grid
            # t_smooth = np.linspace(0, 10, num=200)
            # x_smooth = spline_x(t_smooth)
            # y_smooth = spline_y(t_smooth)
            # z_smooth = spline_z(t_smooth)

            # # 4. Plot the results in 3D
            # fig = plt.figure(figsize=(10, 7))
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(x, y, z, color='red', label='Noisy Mocap/GPS Data')
            # ax.plot(x_smooth, y_smooth, z_smooth, color='blue', label='Fitted 3D Spline Path')
            # ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            # ax.legend()

            plt.show()

            sys.exit()
            
            # identify good segments of the derivative
            threshold = 3
            mask = np.ones(medfilt_subtracted_diff.shape)
            mask[np.where(medfilt_subtracted_diff>=threshold)] = 0
            mask = np.array([mask[0]]+list(mask))
            plt.plot(medfilt_subtracted)
            plt.plot(mask*10)
            plt.show()
            sys.exit()
            
            plt.subplot(3,2,1)
            plt.plot(x_shift_arr)
            plt.ylabel('x')
            plt.subplot(3,2,3)
            plt.plot(y_shift_arr)
            plt.ylabel('y')
            plt.subplot(3,2,5)
            plt.plot(z_shift_arr)
            plt.ylabel('z')
            plt.subplot(3,2,2)
            plt.plot(rad_arr)
            plt.ylabel('rad')
            plt.subplot(3,2,4)
            plt.plot(rad_medfilt)
            plt.ylabel('rad medfilt')
            plt.subplot(3,2,6)
            plt.plot(medfilt_subtracted)
            plt.ylabel('medfilt subtracted')
            plt.show()
            sys.exit()
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

            corr_arr_valid = np.ones(corr_arr.shape)*np.nan
            corr_arr_valid[np.where(valid)] = corr_arr[np.where(valid)]
            x_shift_arr_valid = np.ones(x_shift_arr.shape)*np.nan
            x_shift_arr_valid[np.where(valid)] = x_shift_arr[np.where(valid)]
            y_shift_arr_valid = np.ones(y_shift_arr.shape)*np.nan
            y_shift_arr_valid[np.where(valid)] = y_shift_arr[np.where(valid)]
            z_shift_arr_valid = np.ones(z_shift_arr.shape)*np.nan
            z_shift_arr_valid[np.where(valid)] = z_shift_arr[np.where(valid)]

            if show_plots:
                plt.subplot(2,2,1)
                plt.cla()
                plt.plot(ncorr_arr,label='normalized correlation')
                plt.plot(nrad_arr,label='normalized 3D displacement')
                for k in range(len(invalid)):
                    if invalid[k]:
                        plt.axvline(k,color='r',alpha=0.2)
                plt.legend()
                plt.subplot(2,2,2)
                plt.cla()
                plt.plot(x_shift_arr,'ks',alpha=0.5,label='x shift all')
                plt.plot(x_shift_arr_valid,'ks',alpha=1.0,label='x shift valid')
                plt.legend()
                plt.subplot(2,2,3)
                plt.cla()
                plt.plot(y_shift_arr,'ks',alpha=0.5,label='y shift all')
                plt.plot(y_shift_arr_valid,'ks',alpha=1.0,label='y shift valid')
                plt.legend()
                plt.subplot(2,2,4)
                plt.cla()
                plt.plot(z_shift_arr,'ks',alpha=0.5,label='z shift all')
                plt.plot(z_shift_arr_valid,'ks',alpha=1.0,label='z shift valid')
                plt.legend()
                
                fig_folder = os.path.join(self.output_folder,'figures')
                os.makedirs(fig_folder,exist_ok=True)
                plt.savefig(os.path.join(fig_folder,'regdata_%03d.png'%vidx))
                
                plt.pause(.1)

            self.add(target_volume_filename,
                     x_shift_arr_valid,
                     y_shift_arr_valid,
                     z_shift_arr_valid,
                     corr_arr_valid)
            
    def correct_volumes(self):
        # Determine how large the rendered volume must be.
        # 1. Convert shifts lists to arrays
        # 2. Subtract minimum from each shift array
        # 3. Compute maximum value of shift in each dimension,
        #    and add this to the original volume dimensions

        test = rfunc.get_volume(self.target_volume_filenames[0])
        sy, sz, sx = test.shape
        
        # refactor: implement paging for large volume series that can't be held in RAM
        
        xshifts_vec = np.array(self.xshifts)
        yshifts_vec = np.array(self.yshifts)
        zshifts_vec = np.array(self.zshifts)
        corrs_vec = np.array(self.corrs)
        
        # min-subtract all shifts
        yshifts_vec = yshifts_vec - np.nanmin(yshifts_vec)
        zshifts_vec = zshifts_vec - np.nanmin(zshifts_vec)
        xshifts_vec = xshifts_vec - np.nanmin(xshifts_vec)
        
        # corrected dimensions
        csy = int(sy + np.nanmax(yshifts_vec))
        csz = int(sz + np.nanmax(zshifts_vec))
        csx = int(sx + np.nanmax(xshifts_vec))

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
                corrected_volume[idx+int(y),int(z):int(z+sz),int(x):int(x+sx)] = bscan

            print('Saving %s.'%output_filename)
            np.save(output_filename,corrected_volume)
            self.corrected_filename_dictionary[target_volume_filename] = output_filename

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
        


    
