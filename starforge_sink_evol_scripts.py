# -*- coding: utf-8 -*-
"""
Python module with scripts to analyze the evolution of sinks in the STARFORGE simulations
Created on Tue Mar  1 08:55:09 2022
@author: David Guszejnov
"""

import numpy as np
import os
import glob
import pickle
import get_sink_data
import h5py
from get_sink_data import sinkdata

# #####################################
# #Constants to convert between code units and SI
# #####################################
# pc=3.08567758E16 #pc to m
# AU=1.49598e11 #AU to m 
# yr=31556926.0 #yr to s
# kb=1.38*1E-23 #Boltzmann constant
# mp=1.67*1E-27 #proton mass
# mH2=2.1*mp #assumin it is mostly H2
# msolar=2E30 #solar mass in kg
# msun=msolar #alias
# mu0=np.pi*4E-7 #vacuum permeability, SI
# eps0=8.854E-12 #vacuum permittivity, SI
# G=6.67384*1E-11 #Gravitational constant in SI
# #Conversion constants
# length_unit = pc #pc
# velocity_unit = 1 #m/s
# time_unit=length_unit/velocity_unit #1pc/(1m/s)
# B_field_unit = 1.0 #T
# G_code=G/(pc/msun)

#####################################
#Data acquisition
#####################################


def import_accretion_history(outfilename='accretion_history',folder="output/blackhole_details", redo_pickle=False, selected_ids=None):
    #Import the accretion history, either directly from STARFORGE outputs (i.e., bhswallow_* filles) or from a premade pickle file
    #Returns the accretion_history array as well as its header in 
    #Optionally selected_ids can spacify a list of sink IDs for which we import the history
    accretion_hist_file=folder+'/'+outfilename+'.pickle'
    if (os.path.exists(accretion_hist_file) and not(redo_pickle)):
        print('Loading file '+accretion_hist_file)
        infile=open(accretion_hist_file, 'rb') 
        temp=pickle.load(infile)
        infile.close()
        accretion_history=temp[0];accretion_history_labels=temp[1];temp=0
        print('Loading finished')
    else:  
        accretion_history_labels=np.array(["Time", "ID", "Mass", "Pos[0]", "Pos[1]", "Pos[2]", "Gas ID", "Gas Mass", "dPos[0]", "dPos[1]",\
                       "dPos[2]", "dVel[0]", "dVel[1]", "dVel[2]", "Gas InternalEnergy", "B[0]", "B[1]", "B[2]", "Density"]) 
        BH_files=glob.glob(folder+"/bhswallow_*.txt")     
        accretion_history=None
        if (len(BH_files)==0):
            print("No files found matching pattern "+folder+"/bhswallow_*.txt, trying subfolder...")
            BH_files=glob.glob(folder+"/blackhole_details/bhswallow_*.txt") 
            if (len(BH_files)==0):
                print("No files found matching pattern "+folder+"/blackhole_details/bhswallow_*.txt, exiting...")
                return accretion_history, accretion_history_labels
        for file in BH_files:
            str_arr=np.loadtxt(file,dtype=np.dtype('U25'))
            if(len(str_arr)):
                arr=(str_arr).astype(np.float64)
                if (arr.size==arr.shape[0]):
                    arr=np.reshape(arr,(1,arr.shape[0]))
                #Remove not selected sinks
                if not (selected_ids is None):
                    selected = np.isin(arr[:,1],selected_ids)
                    arr = arr[selected,:]
                    if not len(arr):
                        arr = None
                if accretion_history is None:
                    accretion_history=arr
                elif not (arr is None):
                    accretion_history=np.concatenate((accretion_history, arr), axis=0)
        #sort by time
        sortind = np.lexsort((accretion_history[:,1],accretion_history[:,0])) # Sort by ID, then by time
        accretion_history=accretion_history[sortind,:]
        #Correct for older outputs
        if (len(accretion_history_labels)>accretion_history.shape[1]):
            print("Only %d columns in files while %d was expected"%(accretion_history.shape[1],len(accretion_history_labels)))
            accretion_history_labels = accretion_history_labels[:accretion_history.shape[1]]
        #Pickle
        outfile = open(accretion_hist_file, 'wb') 
        pickle.dump([accretion_history,accretion_history_labels], outfile)
        outfile.close()
    return accretion_history, accretion_history_labels

def import_sink_formation_history(outfilename='sink_formation_history',folder="output/blackhole_details", redo_pickle=False, selected_ids=None):
    #Import the at formation properties of sink particles, either directly from STARFORGE outputs (i.e., bhformation_* filles) or from a premade pickle file
    #Returns the formation properties array as well as its header
    #Optionally selected_ids can spacify a list of sink IDs for which we import the formation properties
    sink_formation_file=folder+'/'+outfilename+'.pickle'
    if (os.path.exists(sink_formation_file) and not(redo_pickle)):
        print('Loading file '+sink_formation_file)
        infile=open(sink_formation_file, 'rb') 
        temp=pickle.load(infile)
        infile.close()
        sink_formation_history=temp[0];sink_formation_labels=temp[1];temp=0
        print('Loading finished')
    else: 
        sink_formation_labels=np.array(["Time", "ID", "Mass", "Pos[0]", "Pos[1]", "Pos[2]", "Vel[0]", "Vel[1]", "Vel[2]",\
                             "B[0]", "B[1]", "B[2]", "Gas InternalEnergy", "Density", "Sound speed", "Size",\
                             "Local Surface Density", "Local Velocity Dispersion", "Distance to closest sink" ])
        BH_files=glob.glob(folder+"/bhformation_*.txt")     
        sink_formation_history=None
        if (len(BH_files)==0):
            print("No files found matching pattern "+folder+"/bhformation_*.txt, trying subfolder...")
            BH_files=glob.glob(folder+"/blackhole_details/bhformation_*.txt") #try standard subfolder
            if (len(BH_files)==0):
                print("No files found matching pattern "+folder+"/blackhole_details/bhformation_*.txt, exiting...")
                return sink_formation_history, sink_formation_labels
        for file in BH_files:
            str_arr=np.loadtxt(file,dtype=np.dtype('U25'))
            if(len(str_arr)):
                arr=(str_arr).astype(np.float64)
                if (arr.size==arr.shape[0]):
                    arr=np.reshape(arr,(1,arr.shape[0]))
                #Remove not selected sinks
                if not (selected_ids is None):
                    selected = np.isin(arr[:,1],selected_ids)
                    arr = arr[selected,:]
                    if not len(arr):
                        arr = None
                if sink_formation_history is None:
                    sink_formation_history=arr
                elif not (arr is None):
                    sink_formation_history=np.concatenate((sink_formation_history, arr), axis=0)
        #sort by time
        sortind = np.argsort(sink_formation_history[:,0]) # Sort by time
        sink_formation_history=sink_formation_history[sortind,:]
        #Pickle
        outfile = open(sink_formation_file, 'wb') 
        pickle.dump([sink_formation_history,sink_formation_labels], outfile)
        outfile.close()
    return sink_formation_history, sink_formation_labels



#####################################
# Utility scripty
####################################

def list_sinks_in_snapshot(snapshot_file):
    F = h5py.File(snapshot_file,'r')
    ids = np.array(F["PartType5/ParticleIDs"],dtype=np.int64)
    m = np.array(F["PartType5/Masses"],dtype=np.float64)
    x = np.array(F["PartType5/Coordinates"],dtype=np.float64)
    v = np.array(F["PartType5/Velocities"],dtype=np.float64)
    #sort by mass
    sortind = np.argsort(m)
    ids = ids[sortind]; m = m[sortind];
    x = x[sortind]; v = v[sortind];
    for i in range(len(ids)):
        print("ID: %d m: %g x: [%g,%g,%g]  v: [%g,%g,%g]"%(ids[i],m[i],x[i,0],x[i,1],x[i,2],v[i,0],v[i,1],v[i,2]))



def start_or_append_in_dict(dictionary, label, value):
    if label in dictionary:
        if not  hasattr(dictionary[label], "__iter__"):
            dictionary[label] = [dictionary[label]] #assume that it already has the first value
        dictionary[label].append(value)
    else:
        dictionary[label] = [value]

def convert_label_and_2Darray_to_dict(array, labels,label_axis=1):
    result_dict = {}
    for i,label in enumerate(labels):
        if label_axis==0:
            result_dict[label] = array[i,:]
        elif label_axis==1:
            result_dict[label] = array[:,i]  
    return result_dict

#####################################
# Single sink evolution
####################################

def get_single_sink_evol(sink_ID, run_folder=None, save_to_folder=None,sink_data_filename="sink_data", snap_name_addition=''):
    if run_folder is None: 
        if save_to_folder is None: 
            run_folder = os.getcwd()
        else:
            run_folder = save_to_folder
    if save_to_folder is None: save_to_folder = run_folder
    #Load the data for all sinks at all times
    outfile_name = save_to_folder+'/'+sink_data_filename+'.pickle'
    if os.path.exists(outfile_name):
        infile=open(outfile_name,'rb')
        snap_data_list = pickle.load(infile)
        infile.close()  
    else:
        snapfiles = glob.glob(run_folder+"/snapshot_*[0-9]"+snap_name_addition+".hdf5")  
        print("Reading sink data from %d snapshots"%(len(snapfiles)))
        snap_data_list = get_sink_data.get_sink_data_in_files(snapfiles,outfolder=save_to_folder,outfilename=sink_data_filename,suppress_output=True,snap_name_addition=snap_name_addition )
    #Sort by time 
    snaptimes = [snapdata.t for snapdata in snap_data_list]
    snap_data_list = np.array(snap_data_list)[np.argsort(snaptimes)]
    #Init for results
    sink_evol = {}
    #Lets find the sink in each snapshot and get its properties
    for snap_data in snap_data_list:
        match = (snap_data.id ==sink_ID)
        if np.any(match):
            ind = np.argmax(match) #get the index
            #let's get the data
            start_or_append_in_dict(sink_evol, "t", snap_data.t)
            start_or_append_in_dict(sink_evol, "x", snap_data.x[ind])
            start_or_append_in_dict(sink_evol, "m", snap_data.m[ind])
            start_or_append_in_dict(sink_evol, "v", snap_data.v[ind])
            start_or_append_in_dict(sink_evol, "formation_time", snap_data.formation_time)
            for label in snap_data.extra_data.keys():
                start_or_append_in_dict(sink_evol, label,snap_data.extra_data[label][ind])
    snap_data_list=0 #force unload
    #Load formation history  
    sink_formation_history,sink_formation_history_labels = import_sink_formation_history(folder=run_folder, selected_ids=[sink_ID])
    sink_formation_history = convert_label_and_2Darray_to_dict(sink_formation_history, sink_formation_history_labels)
    #Load accretion history  
    accretion_history, accretion_history_labels = import_accretion_history(folder=run_folder, selected_ids=[sink_ID])
    accretion_history = convert_label_and_2Darray_to_dict(accretion_history, accretion_history_labels)
    return sink_evol, sink_formation_history, accretion_history







