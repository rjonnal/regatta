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

The manifest contains the registration "recipe". It should be called `manifest.py`. It must contain the following definitions:

1. `target_volume_filenames`: a list (or other iterable) containing the specifiers of the target volumes. These may be folders containing ordered B-scans, or `.npy` files storing entire volumes.

2. `reference_volume_filename`: a specifier for the reference volume, which may be a folder containing B-scans or an `.npy` filename.

3. `output_folder`: where to store intermediate files, corrected volumes, a registered average, etc.

### 
