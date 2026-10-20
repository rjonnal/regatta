## RegATTA (Registration for Advanced Three-dimensional Tomographic Analysis)

RegATTA is a tool for registration of raster-scanned OCT images.

### Overview

Expected data include a **reference** volume and a series of **target** volumes that may include the reference volume. Steps include:

1. Selection of a reference volume by the user. Later versions may include tools for automatic or assisted selection of a reference volume.

2. Specification of the target volumes. Now, these must consist of folders full of B-scans, with each B-scan stored in a separate `.npy` file. The B-scan filenames must correspond to the order in which the B-scans were acquired (i.e., along the slow-scanner axis).

3. The first step in registration is looping through the target volumes. For each target volume, we loop through the B-scans and register each to the reference volume. This produces x, y, and z coordinates for each B-scan, along with the peak value of the DFT-based correlation. This is a time-consuming step, and these values are cached for subsequent runs of the registration script.

4. The second step involves estimating a **trajectory** for each target volume with respect to the reference volume. The trajectory is a 3D curve that describes the motion of the retina with respect to the reference volume. The physical path must be, of course, continuous-valued and continuously differentiable (smooth). The trajectory should be thought of as a discretely sampled version of the physical path, equal in size to the number of B-scans. This step should also provide a confidence value for each point/sample in the trajectory, to permit filtering of registered data according to confidence.

5. The third step uses the trajectory calculated above to correct the target volumes and save corrected copies.

6. Downstream steps may involve averaging the corrected target volumes, segmenting cells from them, etc.

### `manifest.py`

This file should be created and saved in the same folder as `compute_cross_correlations.py`. The manifest contains the registration "recipe". It should be called `manifest.py`. It must contain the following definitions:

1. `target_volume_filenames`: a list (or other iterable) containing the specifiers of the target volumes. These may be folders containing ordered B-scans, or `.npy` files storing entire volumes.

2. `reference_volume_filename`: a specifier for the reference volume, which may be a folder containing B-scans or an `.npy` filename.

3. `output_folder`: where to store intermediate files, corrected volumes, a registered average, etc.

### `registered_volume_series.py`

This package contains the class `RegisteredVolumeSeries`, which does most of the work.

#### `reference_volume`

This variable is held by the `RegisteredVolumeSeries` object. It is initialized with one 3D volume specified by the manifest (i.e., the name of the folder or filename containing the data) and the manifest's output folder, which is used for caching the cross-correlation results. 

#### `RegisteredVolumeSeries`

This object's constructor takes This object handles the the processes of cross-correlation, trajectory estimation, and correction of the target volumes, using the following functions:

1. `compute_cross_correlations` is the member function that performs and caches the B-scan to reference volume cross-correlations. The 3D DFT of the reference volume, used for cross-correlation calculations, is pre-computed at construction. Output of this function is stored in two formats. 

   First, registration results from a single B-scan are cached. Caching is done by hashing the first 128 pixels of the reference volume and then the first 128 pixels of each B-scan to be registered. The two hash codes are concatenated, and the result of the cross-correlation is stored in a file named accordingly. The cache folder is called `.reference_volume_cache` and it is stored in the `output` folder designated in the manifest.
   
   Second, registration results from a target volume are stored in the `output/registration` folder, using slightly mangled filenames from the reference and target volumes.
   
   This is a slight duplication of effort, but for correcting volumes it is more convenient to load the volume-volume registration data rather than assembling the B-scan-to-volume data. The former is also more suitable for separate analysis and visualization.
   
2. `compute_trajectories` loads the registration results from `output/registration` and uses spline fitting (or whatever method we settle on) to correct the trajectories.

3. `correct_volumes` loads the registration results (at present, the uncorrected ones) and writes corrected volumes.

4. `phase_align_volumes` is at present very broken, based on previous architecture with separate reference volume class.


