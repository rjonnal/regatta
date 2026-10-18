## RegATTA (Registration for Advanced Three-dimensional Tomographic Analysis)

RegATTA is a tool for registration of raster-scanned OCT images.

This is an initial release. The minimum working example can be found in `/examples/register_volumes.py`. Sample AO-OCT data for running the script may be found [here](https://www.dropbox.com/scl/fo/ys0tmdk79i6tvkbz28ktp/ALx_5UovCD0_5O_yz5HGTHw?rlkey=h1g2u6ocv9i00g1o7k9s5lngh&dl=0).

The data root folder and output folder must be specified at the top of the script.

Minimally, the script will generate a registered average volume and save it to the output folder.  Several boolean switches are found at the top of the script, that control other outputs:

```python
COMPUTE_FIGURES_OF_MERIT = False
DO_PHASE_CORRECTION = False
PLOT_CORRELATIONS = False
PLOT_PROJECTIONS = False
WRITE_PNGS = False
WRITE_REGISTERED_VOLUMES = False
```

Please see the scripts in `/tests` for additional testing scripts.
