import glob, os

output_folder = '/home/rjonnal/Dropbox/Data/volume_registration/07242026_output'

volume_terms = ['/home/rjonnal/Dropbox/Data/volume_registration/07242026_RJ_OD_t1b_acq5/tacq5_bscans/*',
                '/home/rjonnal/Dropbox/Data/volume_registration/07242026_RJ_OD_t3r_acq8/tacq8_bscans/*']

target_volume_filenames = []

for vt in volume_terms:
    target_volume_filenames = target_volume_filenames + sorted(glob.glob(vt))

target_volume_filenames = sorted(target_volume_filenames)
reference_volume_filename = target_volume_filenames[0]

target_volume_filenames = target_volume_filenames[31:]


