import regatta.registration_functions as rfunc
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from regatta import registered_volume_series
import imageio
import manifest

################ USER PARAMETERS ################

## Script switches
COMPUTE_FIGURES_OF_MERIT = False
DO_PHASE_CORRECTION = False
PLOT_CORRELATIONS = False
PLOT_PROJECTIONS = True
WRITE_PNGS = False
WRITE_REGISTERED_VOLUMES = False

############## END USER PARAMETERS ##############


# register the series to the reference:
rvs = registered_volume_series.RegisteredVolumeSeries(manifest)
rvs.compute_cross_correlations(show_plots=True)
# Use specific_index to work on a particular trajectory; useful
# for skipping the first volume in case it was the reference volume.
# This argument is really only useful for debugging and algorithm development.
rvs.compute_trajectories(show_plots=True,specific_index=3)


rvs.correct_volumes()
sys.exit()


# save the registered average
np.save(os.path.join(manifest.output_folder,'average_volume.npy'),rvs.average_volume)



if COMPUTE_FIGURES_OF_MERIT:

    print('reference volume entropy: ',rvs.compute_entropy(ref))
    print('unregistered average entropy: ',rvs.compute_entropy(np.mean(np.array(vols),axis=0)))
    print('registered average entropy: ',rvs.compute_entropy(rvs.average_volume))

    print('reference volume layer entropy: ',rvs.compute_entropy(ref,bright_layer=True))
    print('unregistered average layer entropy: ',rvs.compute_entropy(np.mean(np.array(vols),axis=0),bright_layer=True))
    print('registered average layer entropy: ',rvs.compute_entropy(rvs.average_volume,bright_layer=True))

    print('reference volume sharpness: ',rvs.compute_sharpness(ref))
    print('unregistered average sharpness: ',rvs.compute_sharpness(np.mean(np.array(vols),axis=0)))
    print('registered average sharpness: ',rvs.compute_sharpness(rvs.average_volume))

    print('reference volume layer sharpness: ',rvs.compute_sharpness(ref,True))
    print('unregistered average layer sharpness: ',rvs.compute_sharpness(np.mean(np.array(vols),axis=0),True))
    print('registered average layer sharpness: ',rvs.compute_sharpness(rvs.average_volume,True))

    print('reference volume contrast: ',rvs.compute_contrast(ref))
    print('unregistered average contrast: ',rvs.compute_contrast(np.mean(np.array(vols),axis=0)))
    print('registered average contrast: ',rvs.compute_contrast(rvs.average_volume))

    print('reference volume layer contrast: ',rvs.compute_contrast(ref,True))
    print('unregistered average layer contrast: ',rvs.compute_contrast(np.mean(np.array(vols),axis=0),True))
    print('registered average layer contrast: ',rvs.compute_contrast(rvs.average_volume,True))

if PLOT_CORRELATIONS:

    pre_corrs = []
    for k in range(len(vols)):
        pre_corrs.append(rvs.compute_correlation(ref,vols[k]))

    post_corrs = []
    for k in range(len(vols)):
        post_corrs.append(rvs.compute_correlation(rvs.corrected_volumes[refidx],rvs.corrected_volumes[k]))

    plt.figure()
    plt.plot(range(len(vols)),pre_corrs,'gs',label='before registration')
    plt.plot(range(len(vols)),post_corrs,'ro',label='after registration')
    plt.xlabel('volume index')
    plt.ylabel('Pearson correlation with volume %d (OCT magnitude only)'%refidx)
    plt.legend()
    plt.xticks(range(0,20,5))
    plt.savefig(os.path.join(manifest.output_folder,'volume_correlation.png'),dpi=300)


if DO_PHASE_CORRECTION:
    # phase-align the volumes
    rvs.phase_align_volumes()
    np.save(os.path.join(manifest.output_folder,'average_volume_phase_corrected.npy'),rvs.average_volume)

if PLOT_PROJECTIONS:
    # visualize the result somehow:
    rad = 1
    aavg = np.abs(rvs.average_volume)
    aref = np.abs(ref)
    yidx = np.nanargmax(np.nanmean(aavg,(1,2)))
    zidx = np.nanargmax(np.nanmean(aavg,(0,2)))
    xidx = np.nanargmax(np.nanmean(aavg,(0,1)))
    plt.figure(figsize=(9,6))
    plt.subplot(2,3,1)
    plt.imshow(np.nanmean(aref[yidx-rad:yidx+rad+1,:,:],axis=0))
    plt.subplot(2,3,2)
    plt.imshow(np.nanmean(aref[:,zidx-rad:zidx+rad+1,:],axis=1))
    plt.subplot(2,3,3)
    plt.imshow(np.nanmean(aref[:,:,zidx-rad:zidx+rad+1],axis=2).T)
    plt.subplot(2,3,4)
    plt.imshow(np.nanmean(aavg[yidx-rad:yidx+rad+1,:,:],axis=0))
    plt.subplot(2,3,5)
    plt.imshow(np.nanmean(aavg[:,zidx-rad:zidx+rad+1,:],axis=1))
    plt.subplot(2,3,6)
    plt.imshow(np.nanmean(aavg[:,:,zidx-rad:zidx+rad+1],axis=2).T)
    plt.show()

if WRITE_PNGS:
    # save the registered, averaged frames as PNG files
    temp = np.zeros(rvs.average_volume.shape)
    temp[...] = rvs.average_volume[...]
    temp = np.abs(temp)
    temp[np.where(np.isnan(temp))] = np.nanmin(temp)
    temp = np.round((temp-np.min(temp))/(np.max(temp)-np.min(temp))*255).astype(np.uint8)
    png_folder = os.path.join(manifest.output_folder,'average_volume_png')
    os.makedirs(png_folder,exist_ok=True)
    plt.figure(figsize=(6,6))
    
    for k in range(temp.shape[0]):
        outfn = os.path.join(png_folder,'%05d.png'%k)
        im = temp[k,:,:]
        plt.cla()
        plt.imshow(im,cmap='gray')
        plt.title('B-scan %03d'%k)
        imageio.imwrite(outfn,im)
        print('Wrote %s.'%outfn)
        plt.pause(0.01)

if WRITE_REGISTERED_VOLUMES:
    # save the corrected volumes:
    registered_folder = os.path.join(manifest.output_folder,'registered_volumes')
    os.makedirs(registered_folder,exist_ok=True)

    for idx,vol in enumerate(rvs.corrected_volumes):
        volume_filename = os.path.join(registered_folder,'%05d_registered.npy'%idx)
        np.save(volume_filename,vol)
        print('Saving registered volume %s.'%volume_filename)
        
plt.show()
