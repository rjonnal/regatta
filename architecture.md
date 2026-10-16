## Manifest

The manifest contains the registration "recipe". It should be called `manifest.py`. It must contain the following definitions:

1. `target_volume_filenames`: a list (or other iterable) containing the specifiers of the target volumes. These may be folders containing ordered B-scans, or `.npy` files storing entire volumes.

2. `reference_volume_filename`: a specifier for the reference volume, which may be a folder containing B-scans or an `.npy` filename.

3. `output_folder`: where to store intermediate files, corrected volumes, a registered average, etc.

## Architecture

