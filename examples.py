# -*- coding: utf-8 -*-
"""
Example calls for STARFORGE sink analysis

@author: David Guszejnov
"""

import numpy as np
import time

# #####################################
# #Constants to convert between code units and SI
# #####################################
pc=3.08567758E16 #pc to m
AU=1.49598e11 #AU to m 
yr=31556926.0 #yr to s
kyr=1e3*yr; Myr=1e6*yr
kb=1.38*1E-23 #Boltzmann constant
mp=1.67*1E-27 #proton mass
mH2=2.1*mp #assuming it is mostly H2
msolar=2E30 #solar mass in kg
msun=msolar #alias
G=6.67384*1E-11 #Gravitational constant in SI
#Conversion constants
length_unit = pc #pc
velocity_unit = 1 #m/s
time_unit=length_unit/velocity_unit #1pc/(1m/s)
B_field_unit = 1.0 #T
G_code=G/(pc/msun)



#####################################
# #Get a list of sink particles in a snapshot
#####################################
from starforge_sink_evol_scripts import list_sinks_in_snapshot
list_sinks_in_snapshot("example_STARFORGE_output/snapshot_200_stars.hdf5")
time.sleep(2)

####################################################
# Evolution of a single sink particle
###################################################

#Plot the evolution of a single sink
from matplotlib import pyplot as plt
from starforge_sink_evol_scripts import get_single_sink_evol
from get_sink_data import sinkdata
from GDplot import plot


chosen_sink_id = 172322
sink_evol, sink_formation_history, accretion_history = get_single_sink_evol(chosen_sink_id, run_folder='example_STARFORGE_output', save_to_folder="example_STARFORGE_output",sink_data_filename="M2e2_full_physics_2e6", snap_name_addition="_stars")
#Let's get the at formation properties of the sink
print("At formation properties of sink:")
for label in sink_formation_history.keys():
    print("\t",label, sink_formation_history[label][0])
t0 = sink_formation_history['Time'][0] #formation time
time.sleep(2)

#The accretion_history contains every single accretion event for the chosen sink, this data is much finer than the coarse sink data we can get from snapshots (i.e., in sink_evol)
print("Keys in accretion_history", accretion_history.keys())
dt_acc = accretion_history['Time'] - t0 
plot(dt_acc*time_unit/kyr, np.log10(accretion_history['Mass']),'','Time since formation [kyr]', r'Log Mass [$\mathrm{M}_\odot$]','example_plots/mass_evol')

#Most sink properties are not stored in the accretion history files so we get them from the snapshots themselves (i.e., they are in sink_evol)
print( "Snapshot spacing is %g kyr"%(np.mean(np.diff(sink_evol['t']))*time_unit/kyr) )
print("Keys in sink_evol", sink_evol.keys())
dt = sink_evol['t'] - t0
plot(dt*time_unit/kyr, np.log10(sink_evol['ProtoStellarRadius_inSolar']),'','Time since formation [kyr]', r'Log R [$\mathrm{R}_\odot$]','example_plots/radius_evol')
plot(dt*time_unit/kyr, np.log10(sink_evol['StarLuminosity_Solar']),'','Time since formation [kyr]', r'Log L [$\mathrm{L}_\odot$]','example_plots/luminsoity_evol')
time.sleep(2)


####################################################
# Look at global sink mass statistics (i.e. IMF) 
###################################################
from starforge_IMF_scripts import model_comparison

#We are comparing the stellar mass statistics between runs
filenames = ['M2e2_full_physics_2e6'] #filenames of sink_data files, created with get_sink_data.get_sink_data_in_files
runnames = [r'$M = 2\times 10^2\,M_\mathrm{\odot}$, $N = 2\times 10^6$'] #label in the plots for each run
#Parameters
completeness_limit = 0.01; min_particle_num = completeness_limit/1e-4 #using that the example here has a resolution of 1e-4 msun
logmassbin_edges = np.concatenate((np.linspace(-3.0,0,19),np.linspace(0.25,1,3),np.linspace(1.5,3,4))) #mass bins for the IMF
#The model_comparison is a bit of a frankenstein's monster, it has lots of parameters to adjust the figures, but here is a simple version
model_comparison(filenames, data_folder='example_STARFORGE_output', target_SFE='Max', modelnames=runnames, label='',\
                 min_particle_num=min_particle_num, logmassbin_edges=logmassbin_edges,plot_obs_IMF=True,\
                 overtextcoord=[0.02,0.03], xlim=[-2.3, 1.0])
     


 

# from matplotlib import pyplot as plt   
# sink_evol, sink_formation_history, accretion_history = get_single_sink_evol(2020819, run_folder='C:/Users/gusze/Desktop/temp/BH_files', save_to_folder="D:\Work\Projects\GMC Sim\Analyze\sinkdata",sink_data_filename="M2e5_C_M_J_RT_W_2e7")
# t0 = sink_formation_history['Time'][0]
# dt = accretion_history['Time'] - t0 
# plt.plot(dt*pc/yr/1000, np.log10(accretion_history['Mass']))







