import regatta.registration_functions as rfunc
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from regatta import registered_volume_series
import imageio


################ USER PARAMETERS ################

## Script switches

# data for this test can be downloaded from the link below:
# https://www.dropbox.com/scl/fo/ys0tmdk79i6tvkbz28ktp/ALx_5UovCD0_5O_yz5HGTHw?rlkey=h1g2u6ocv9i00g1o7k9s5lngh&dl=0

# point 'root_folder' to the location of the data downloaded above
# provide a folder name for storage of output
# warning: the script will create the output folder and overwrite any existing
# contents
root_folder = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct'
output_folder = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct_output'

############## END USER PARAMETERS ##############


# root_folder = 'C:/bscans_aooct'

volume_filenames = sorted(glob.glob(os.path.join(root_folder,'*')))

vols = [rfunc.get_volume(fn,prefix='') for fn in volume_filenames]

# use a volume from the middle of the series as a reference:
refidx = len(vols)//2

output_folder = os.path.join(output_folder,'refidx_%05d'%refidx)
os.makedirs(output_folder,exist_ok=True)

# get the reference volume
ref = vols[refidx]
reference_data = ref

# register the series to the reference:
rvs = registered_volume_series.RegisteredVolumeSeries(ref)

for erms in np.arange(0,6):
    rvs.register_volumes(vols,error_rms=erms)
    print(erms,rvs.compute_entropy(rvs.average_volume,n_bins=64,bright_layer=True))

