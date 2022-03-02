#!/usr/bin/env python
"""
Usage:
get_sink_data.py <files> ... [options]

Options:
    -h --help               Show this screen.
   --sinktype=<n>           Particle type for stars [default: 5]
   --save_local             Flag, if set the output is saved in the current directory
   --outfolder=<name>       Name of output folder, defaults to snapshot folder [default: none]
   --outfilename=<name>     Name of output file [default: sink_data]
   --suppress_output        Flag, if set all outputs are suppressed
"""

#Example
#python get_sink_data.py output/snapshot_*[0-9].hdf5 --save_local

from docopt import docopt
import pickle
import numpy as np
from load_from_snapshot import load_from_snapshot
import re
import os
from natsort import natsorted
#########
#Code to enable suppression of stdout
import contextlib
import sys
class DummyFile(object):
    def write(self, x): pass
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
##########

class sinkdata:
    def __init__(self,m,x,v,ids,t,snapnum,numpart_total, formation_time):
        self.x = x
        self.v = v
        self.m = m
        self.id = ids
        self.t = t
        self.formation_time = formation_time
        self.snapnum = snapnum
        self.numpart_total = numpart_total
        self.extra_data={}
        
    def add_extra_data(self,name,value):
        self.extra_data[name] = value

def getsnapnumandfolder(filename):
    if os.path.isdir(filename):
        namestring="snapdir"
    else:
        namestring="snapshot"
    snapnum = int(re.search(namestring+'_\d*', filename).group(0).replace(namestring+'_',''))
    datafolder=(filename.split(namestring+"_")[0])
    return snapnum,datafolder
    
def weight_median(x,weights):
    weightsum=np.sum(weights)
    if weightsum:
        if (np.size(x)==1):
            if hasattr(x, "__iter__"):
                return x[0]
            else:
                return x
        else:
            #reorder x
            sortind=np.argsort(x)
            cdf=np.cumsum(weights[sortind])/np.sum(weights)
            median_index=np.argmax(cdf>=0.5)
            return x[sortind[median_index]]
    else:
        return 0

def get_sink_data_in_files(files,sinktype=5,outfolder='none',outfilename='sink_data',save_local=False,suppress_output=False,snap_name_addition='' ):
    data=[]
    _,datafolder=getsnapnumandfolder(files[0])
    if save_local:
        outfolder_in='./'
    else:
        if outfolder=='none':
            outfolder_in=datafolder
        else:
            outfolder_in=outfolder
    filename=outfolder_in+'/'+outfilename+'.pickle'

    for f in files:
        if not suppress_output: print(f)
        snapnum,datafolder=getsnapnumandfolder(f)
        time = load_from_snapshot("Time",0,datafolder,snapnum,name_addition=snap_name_addition)
        numpart_total = load_from_snapshot("NumPart_Total",0,datafolder,snapnum,name_addition=snap_name_addition)
        if (numpart_total[sinktype]):
            ms=load_from_snapshot('Masses',sinktype,datafolder,snapnum,name_addition=snap_name_addition)
            xs=load_from_snapshot('Coordinates',sinktype,datafolder,snapnum,name_addition=snap_name_addition)
            vs=load_from_snapshot('Velocities',sinktype,datafolder,snapnum,name_addition=snap_name_addition)
            ids=load_from_snapshot('ParticleIDs',sinktype,datafolder,snapnum,name_addition=snap_name_addition)
            formation_time=load_from_snapshot('StellarFormationTime',sinktype,datafolder,snapnum,name_addition=snap_name_addition)
            s = sinkdata(ms,xs,vs,ids, time, snapnum, numpart_total,formation_time)
            m50 = weight_median(ms,ms)
            if not suppress_output: print("\t %d sink particles, masses: %g total, %g mean %g median %g mass weighted median %g max"%(len(ms),np.sum(ms),np.mean(ms),np.median(ms),m50, np.max(ms)))
            #Get extra data
            with nostdout():
                keys = load_from_snapshot("keys",sinktype,datafolder,snapnum,name_addition=snap_name_addition)
            basic_data=['Masses','Coordinates','Velocities','ParticleIDs','StellarFormationTime']
            stuff_to_skip=[]
            for key in keys:
                if not ( (key in basic_data) or (key in stuff_to_skip) ): #only those we have not already stored and we actually want
                    val = load_from_snapshot(key,sinktype,datafolder,snapnum,name_addition=snap_name_addition)
                    s.add_extra_data(key,val) #add data to class, can be read with val method
            data.append(s)
        else:
            if not suppress_output: print("\t No sinks present")
    if not suppress_output: print("Saving to "+filename)
    outfile = open(filename, 'wb') 
    pickle.dump(data, outfile)
    outfile.close()
    return data

if __name__ == "__main__":
    arguments = docopt(__doc__)
    files = natsorted(arguments["<files>"])
    sinktype = int(arguments["--sinktype"])
    outfolder=arguments["--outfolder"]
    outfilename=arguments["--outfilename"]
    save_local=arguments["--save_local"]
    suppress_output=arguments["--suppress_output"]
    get_sink_data_in_files(files,sinktype,outfolder,outfilename,save_local,suppress_output)

    
