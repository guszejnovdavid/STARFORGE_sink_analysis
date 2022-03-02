# STARFORGE_sink_analysis
Scripts to analyze sink particle (i.e., star) data in the [STARFORGE simulations](http://starforge.space/).
See `examples.py` for some working examples with the provided STARFORGE outputs from a small (M = 200 Msun, R = 1pc ) run with all physics enabled.
You can get more STARFORGE star data from [this repository](https://github.com/mikegrudic/StarforgeFullPhysics).

## Installation
Download the repository and import the necessary scripts into your analysis routine.

**Required Python packages**: `numpy`,`h5py`,`docopt`,`natsort` as well as `palettable` for plotting.


## Usage

### Get a list of sink particles in a snapshot
To get a list of sinks and their basic properties (ID, mass, position, velocity) from an HDF5 snapshot use
```
from starforge_sink_evol_scripts import list_sinks_in_snapshot
list_sinks_in_snapshot("path_to_snapshot/snapshot_x.hdf5")
```

### Evolution of a single star (sink)
To get the full evolution and accretion history for a single star use 
```
from starforge_sink_evol_scripts import get_single_sink_evol
from get_sink_data import sinkdata
sink_evol, sink_formation_history, accretion_history =  get_single_sink_evol(sink_ID, run_folder=None, save_to_folder=None)
```
where `sink_ID` is the unique particle ID of the star you are interested in, `run_folder` is where the STARFORGE outputs (i.e., snapshots, accretion histories) are. The script will also generate pickle files that contain the total sink evolution and accretion histories, which are saved in a pickle file in `save_to_folder` (if these are present the script does not require the actual STARFORGE outputs any more). By default the current working directory is used for both.

The scrip returns `sink_evol`, `sink_formation_history`, `accretion_history`, which are dictionaries containing the evolution of stellar properties (in each snapshot), at formation properties and accretion history for the star `with sink_ID`.

### Evolution of global stellar properties
We can use `model_comparison` to create a set of plots that show the evolution of various star formation metrics (i.e., star formation rate/efficiency, mean/median/weighted-median stellar mass, effective IMF slope) and the initial mass function. 
```
from starforge_IMF_scripts import model_comparison
filenames = ['filename1', 'filename2'] #filenames of sink_data files, created with get_sink_data.get_sink_data_in_files
runnames = [r'$M = 2\times 10^2\,M_\mathrm{\odot}$, $N = 2\times 10^6$'] #labels in the plots for each run
model_comparison(filenames, data_folder='folder_for_files', modelnames=runnames)
```
Note that model_comparison has a lot of optional parameters that can be used to adjust the figures, for a basic call see `examples.py`.

