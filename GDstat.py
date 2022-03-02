#Statistics and other routines
"""
Created on Wed Jan 16 09:22:23 2019

@author: gusze
"""

import numpy as np
from scipy import stats #scipy stats routines
from scipy import optimize

def moving_avg(x,N=5,mode='valid'):
    valid_ind1=(int)((N-1)/2);valid_ind2=len(x)-(int)((N-1)/2)
    x_avg=np.convolve(x, np.ones((N,))/N, mode=mode)
    return x_avg, valid_ind1, valid_ind2

def func_on_list_elements(array, func):
        return [func(l) for l in array]

def npmean(x):
    return np.mean(x)

def npmedian(x):
    return np.median(x)

def inside_bounds_indices(y,ymin=None,ymax=None):
    y = np.array(y)
    ind = np.full(len(y),True)
    if not (ymin is None): ind = ind & (y>=ymin)
    if not (ymax is None): ind = ind & (y<=ymax)
    return np.arange(len(y))[ind] 

def list_from_classlist(class_list,elementname,indices=[]):
    out=[]
    if len(indices):
        for i in indices:
            out.append(getattr(class_list[i], elementname))
    else:
        for c in class_list:
            out.append(getattr(c, elementname))
    return out

def withinrange(x,minval,maxval):
    'Returns which elements are witihin the prescribed range'
    return (x <= maxval) & (x >= minval)


def weight_avg(data,weights):
    'Weighted average'
    if weights is None:
        return np.mean(data)
    else:
        weightsum=np.sum(weights)
        if weightsum:
            return np.sum(data*weights)/weightsum
        else:
            return 0*data

def weight_mean(data,weights):
    return weight_avg(data,weights)

def weight_std(data,weights):
    'Weighted standard deviation'
    weightsum=np.sum(weights)
    if weightsum and (len(data)>1):
        return np.sqrt( np.sum((data**2)*weights)/np.sum(weights) - (weight_avg(data,weights)**2) )
    else:
        return 0*data

def weight_median(data,weights):
    'Weighted median'
    weightsum=np.sum(weights)
    if weightsum:
        if (np.size(data)==1):
            if hasattr(data, "__iter__"):
                return data[0]
            else:
                return data
        else:
            #reorder data
            sortind=np.argsort(data)
            cdf=np.cumsum(weights[sortind])/np.sum(weights)
            median_index=np.argmax(cdf>=0.5)
            return data[sortind[median_index]]
    else:
        return 0
 
def weight_mean_within_range(data,weights,limits):
    ind = withinrange(data,limits[0],limits[1])
    if weights==0:
        weights = np.ones_like(data)
    if np.any(ind):
        return weight_mean(data[ind],weights[ind])
    else:
        return 0
    

def cart_to_spherical(xyz): #taken from stackoverflow
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
   
def mom_inertia_periodic_callfunc(center,x,m,periodic):
    dx = (x - center) % periodic #relative coorddinates from center
    dx[dx>periodic/2] -= periodic
    #dx[dx<-periodic/2] += periodic #unnecessary
    return np.sum((dx**2)*m[:,None])
    

def find_periodic_CoM(x_orig,m,periodic, Ntries=5):
    #Find the center of mass in a periodic box
    CoM = np.sum(x_orig*m[:,None],axis=0)/np.sum(m) #using no periodicity
    if periodic>0:
        sol_list=[]
        x = x_orig % periodic #enforce periodicity just to be sure
        for i in range(Ntries): #trying different starting points to avoid issues with weird geometries
            shift = np.sum( ((x+i/Ntries*periodic) % periodic)*m[:,None],axis=0)/np.sum(m) #let's start with naive center of mass estimate          
            sol = optimize.minimize(mom_inertia_periodic_callfunc, shift, args=(x,m,periodic),\
                                bounds=[(0,periodic),(0,periodic),(0,periodic)]) #find best shift
            sol_list.append(sol)
        CoM = sol_list[np.argmin([sol['fun'] for sol in sol_list])]['x']
    return CoM

def weight_percentile(data,weights,p):
    'Weighted percentile calculation'
    weightsum=np.sum(weights)
    if weightsum:
        if (np.size(data)==1):
            if hasattr(data, "__iter__"):
                return data[0]
            else:
                return data
        else:
            #reorder data
            sortind=np.argsort(data)
            cdf=np.cumsum(weights[sortind])/np.sum(weights)
            if(p>1.0):
                p /= 100.0
                print("Large percentile value gven, converting %g to %g"%(p*100,p))
            percentile_index=np.argmax(cdf>=p)
            return data[sortind[percentile_index]]
    else:
        return 0
def edge_to_bin(array):
    return (array[:-1]+array[1:])/2.0

def bin_to_edge(array):
    dx1=(array[1]-array[0])
    dx2=(array[-1]-array[-2])
    mean_array=(array[:-1]+array[1:])/2.0
    return np.append(np.array([array[0]-dx1/2.0]),np.append(mean_array,np.array([array[-1]+dx2/2.0])))

def CDF_at_value_from_PDF(bins,pdf,cdf_val):
    cdf=np.cumsum(pdf)/np.sum(pdf) #normalized CDF
    index=np.argmax(cdf>=cdf_val)
    return bins[index]

def CDF_at_value_from_PDF_interpol(bins,pdf,cdf_val):
    cdf=np.cumsum(pdf)/np.sum(pdf) #normalized CDF
    edges=bin_to_edge(bins)
    index=np.argmax(cdf>=cdf_val)
    if index>1:
        weight = 1.0-((cdf[index]-cdf_val)/(cdf[index]-cdf[index-1]))**2
        return ( edges[index+1]*weight + (1-weight)*edges[index] )
    else:
        return bins[index]

def make_CDF(data,start_from_high=False,normed=True, weights=None):
    sortindex=np.argsort(np.array(data).flatten())
    sortdata=(np.array(data).flatten())[sortindex]
    if not (weights is None):
        if start_from_high:
            cdf=np.flip(np.cumsum(np.flip(weights[sortindex])))
        else:
            cdf=np.cumsum(weights[sortindex])
    else:
        if start_from_high:
            cdf=np.flip(np.cumsum(np.ones(len(sortdata)))) #no need to flip inside
        else:
            cdf=np.cumsum(np.ones(len(sortdata)))
    if normed:
        cdf=cdf/np.max(cdf) #normalized CDF
    return sortdata,cdf

def CDF_at_value_from_data(data,cdf_val, start_from_high=False):
    sortdata=np.sort(np.array(data).flatten())
    cdf=np.cumsum(np.ones(len(sortdata)))
    if start_from_high:
        cdf=np.flip(cdf)
    cdf=cdf/np.max(cdf) #normalized CDF
    index=np.argmax(cdf>=cdf_val)
    return sortdata[index]



def normalize_array(array):
    arrmax=np.max(array)
    arrmin=np.min(array)
    return (array-arrmin)/(arrmax-arrmin)

def find_peak_interval_in_histogram(hist,threshold,vals=None,nobreak=False):
    'Finds the interval around the maximum of the array where the value is at least threshold*max'
    maxind=np.argmax(hist);leftind=maxind;rightind=maxind;
    if vals is None:
        vals=range(len(hist))
    #To the right
    for i in range(maxind,len(hist)):
        if (hist[i]>=threshold*hist[maxind]):
            rightind=i
        else:
            if not(nobreak):
                break #Stop if it went too far
    for i in range(maxind,-1,-1):
        if (hist[i]>=threshold*hist[maxind]):
            leftind=i
        else:
            if not(nobreak):
                break #Stop if it went too far
    if (leftind==maxind) and (maxind>0): leftind=maxind-1;
    if (rightind==maxind) and (maxind<(len(hist)-1)): rightind=maxind+1;
    return [vals[leftind],vals[rightind]]

def polyfit_with_error(x,y,degree,yerr=None,verbose=False):
    if yerr is None:
        w=None; cov=True
    else:
        w=1/yerr; cov='unscaled'
    p,V = np.polyfit(x,y,degree,cov=cov, w=w)
    p_err = np.array([np.sqrt(V[i][i]) for i in range(degree+1)])
    if verbose:
        for i in range(degree+1):
            print("x_%d: %g +/- %g"%(i,p[i], p_err[i]) )
    return p, p_err, V
    
    

def average_list(x,y,N_avg):
    x=np.array(x);y=np.array(y);
    N=len(x)
    mod=N%N_avg
    N_new=N/N_avg
    x_avg=np.zeros(N_new);y_avg=np.zeros(N_new);y_err=np.zeros(N_new);
    for i in range(N_new):
        ind=range(i*N_avg,(i+1)*N_avg)
        x_avg[i]=np.mean(x[ind])
        y_avg[i]=np.mean(y[ind])
        y_err[i]=np.std(y[ind])
    if (mod==1 or (mod>0 and mod<(N_avg/5))):
        if (N_avg<4):
            x_avg=np.append(x_avg,x[-1])
            y_avg=np.append(y_avg,y[-1])
            y_err=np.append(y_err,0)
        else:
            ind1=range(N-N_avg/2-1,N)
            ind2=range(N-N_avg,N-N_avg/2-1)
            x_avg[-1]=np.mean(x[ind2])
            y_avg[-1]=np.mean(y[ind2])
            y_err[-1]=np.std(y[ind2])
            x_avg=np.append(x_avg,np.mean(x[ind1]))
            y_avg=np.append(y_avg,np.mean(y[ind1]))
            y_err=np.append(y_err,np.std(y[ind1]))
    elif (mod>0):
        ind=range(N-mod,N)
        x_avg=np.append(x_avg,np.mean(x[ind]))
        y_avg=np.append(y_avg,np.mean(y[ind]))
        y_err=np.append(y_err,np.std(y[ind]))
    return x_avg,y_avg,y_err

def average_inbin_list(x,y,binedges, weights=None):
    if weights is None:
        weights = np.ones_like(x)
    Nbin=len(binedges)-1
    x=np.array(x);y=np.array(y);
    x_avg=np.zeros(Nbin);y_avg=np.zeros(Nbin);y_err=np.zeros(Nbin);y10=np.zeros(Nbin);y90=np.zeros(Nbin);
    for i in range(Nbin):
        ind= ( (x>=binedges[i]) & (x<binedges[i+1]))
        if np.any(ind):
            x_avg[i]=weight_mean(x[ind],weights[ind])
            y_avg[i]=weight_mean(y[ind],weights[ind])
            y_err[i]=weight_std(y[ind],weights[ind])
            y10[i] = weight_percentile(y[ind],weights[ind],0.1)
            y90[i] = weight_percentile(y[ind],weights[ind],0.9)
    return x_avg,y_avg,y_err,y10,y90

def binned_stats(x,y,statistic='mean', bins=10, range=None):
    #Bins values by x, then for each bin calculates the statistics for the y values
    if hasattr(bins, "__iter__"):
        #in case it is really bins
        newbins=bin_to_edge(bins)
    else:
        newbins=bins
    bin_func_val, bin_edges, binnumber = stats.binned_statistic(x, y, statistic=statistic, bins=newbins, range=range)
    return bin_func_val, edge_to_bin(bin_edges), (binnumber-1)

def hist_equal_pop_bins(data,nbin, density=False,weights=None):
    if (len(data)<(nbin+1)):
        nbin=len(data)-1
    if nbin==0:
        return np.ones(1), np.array([data[0]-1e-10,data[0]+1e-10])
    elif nbin==1:
        return np.array([len(data)]), np.array([np.min(data),np.max(data)])
    else:    
        sortdata=np.sort(np.array(data))
        percentile_vals=np.linspace(0,100.0,num=(nbin+1))
        edges=np.percentile(sortdata,percentile_vals)
        if (weights is None):
            hist, _=np.histogram(sortdata, bins=edges, density=density,weights=weights)
        else:
            hist, _=np.histogram(sortdata, bins=edges, density=density)
        return hist, edges

def advanced_binning(data,target_size, min_bin_pop=0,max_bin_size=1e100,dx=1e-10,allow_empty_bins=True):
    #Give bin edges of target size while ensuring that we have at least min_bin_pop elements. No bins can be bigger than max_bin_size, even if the population is smaller than min_bin_pop but also at least 1  
    sortdata=np.sort(np.array(data)); N = len(sortdata)
    edges = [sortdata[0]-dx];  lastind = 0; #init
    for i,x in enumerate(sortdata):
        if i>0:
            bin_pop = i-lastind
            bin_val = (sortdata[i-1]+sortdata[i])/2
            bin_size = bin_val-edges[-1]
            if allow_empty_bins:
                while bin_size>2*max_bin_size:
                    edges.append(edges[-1]+max_bin_size); lastind=i
                    bin_size = bin_val-edges[-1] 
            if (i>N-min_bin_pop) and ( (sortdata[-1]-bin_val)<target_size ):
                edges.append(sortdata[-1])
                break
            elif ( (bin_pop>=min_bin_pop) and (bin_size>=target_size) ) or (bin_size>max_bin_size):
                edges.append(bin_val)
                lastind = i
    if edges[-1]<=sortdata[-1]:  edges[-1] = sortdata[-1]+dx
    return np.array(edges)

def hist2D_with_stat(xvals,yvals,x_edges=0, y_edges=0, xbins=10,ybins=10,weights=0,\
                             stat_val_function='weight_median', stat_err_function_plus='weight_percentile',\
                             stat_err_function_minus='weight_percentile',errplus_param=0.75,errminus_param=0.25,\
                             xpercentile_limits=[0,100],ypercentile_limits=[0,100], normalize=True):
    #Creates a 2D histogram as well as calculating the stat_val_function and stat_err_function in each x bin
    #Get bin edges
    if not hasattr(x_edges, "__iter__"):
        if hasattr(xbins, "__iter__"):
            x_edges=bin_to_edge(xbins)
        else:
            x_edges=np.linspace(np.percentile(xvals,xpercentile_limits[0]),np.percentile(xvals,xpercentile_limits[1]),xbins)
            xbins=edge_to_bin(x_edges)
    else:
        xbins=edge_to_bin(x_edges)
    if not hasattr(y_edges, "__iter__"):
        if hasattr(ybins, "__iter__"):
            y_edges=bin_to_edge(ybins)
        else:
            y_edges=np.linspace(np.percentile(yvals,ypercentile_limits[0]),np.percentile(yvals,ypercentile_limits[1]),ybins)
            ybins=edge_to_bin(y_edges)
    else:
            ybins=edge_to_bin(y_edges)    
    #Check weights, if not given, use equal
    if (not hasattr(weights, "__iter__")):
        weights=np.ones(len(xvals))
    #make 2D histogram
    hist, _, _=np.histogram2d(xvals,yvals, bins=[x_edges,y_edges], density=normalize, weights=weights)
    #first calculate stat_val_function for y values in each bin
    binnumber = np.digitize(xvals, x_edges)
    binnumber=np.array(binnumber)-1
    #calculate stat_err_function for each bin
    y_val_plus=np.zeros(len(xbins))
    y_val_minus=np.zeros(len(xbins))
    y_stat_val=np.zeros(len(xbins))
    for i in range(len(xbins)):
        ind=(binnumber==i)
        if (np.sum(ind)>0):
            y_stat_val[i]=globals()[stat_val_function](yvals[ind], weights[ind])
            y_val_plus[i]=globals()[stat_err_function_plus](yvals[ind], weights[ind],errplus_param)
            y_val_minus[i]=globals()[stat_err_function_minus](yvals[ind], weights[ind],errminus_param)
        else:
            y_stat_val[i]=np.nan
            y_val_plus[i]=y_stat_val[i]
            y_val_minus[i]=y_stat_val[i]
    return hist,y_stat_val,y_val_plus,y_val_minus,xbins,ybins

def MC_stat_func_guess(data, weights=None,stat_function='weight_median',stat_function_param1=None,\
                       stat_function_param2=None,stat_function_param3=None, N_MC=100,fixed_sum=None, fixed_sum_err_tol=0.1,\
                       max_iter=100,confidence=0):
    if weights is None:
        weights=np.full(len(data),1.0/len(data))
    else:
        weights = weights/np.sum(weights)
    #Generate distribution for index
    random_variable = stats.rv_discrete(values=(np.arange(len(data)),weights))
    #Create new realizations and calculate the function value
    vals = []
    N_size = len(data)
    for i in range(N_MC):
        #We create realizations with the same number of samples
        chosen_ind=random_variable.rvs(size=N_size)
        if not (fixed_sum is None):
            if hasattr(fixed_sum, "__iter__"):
                #it is not the number of samples that should be fixed but the sum of the supplied vector (e.g. total mass of objects instead of number)
                mass_target=np.sum(fixed_sum)
            else:
                mass_target = fixed_sum
            mass_sum = np.sum(data[chosen_ind]); 
            i=0
            while ( (i<max_iter) and (np.abs(1.0-mass_sum/mass_target)>fixed_sum_err_tol) ):
                #we need to add/remove to get closer to th target mass
                excess_N = np.max([1,int(np.abs((mass_sum/mass_target-1.0)*N_size*0.5))])
                if (mass_sum>mass_target):
                    if (excess_N>len(chosen_ind)):
                        excess_N = int(len(chosen_ind)/2)
                    chosen_ind=chosen_ind[:-excess_N]
                else:
                    chosen_ind=np.append(chosen_ind,random_variable.rvs(size=excess_N))
                mass_sum = np.sum(data[chosen_ind])
                i+=1
                #print(i,mass_sum,mass_target,N_size,excess_N)
        #the realization we want
        realization = data[chosen_ind]            
        if (stat_function=='weight_median'):
            val = globals()[stat_function](realization,realization)
        elif (stat_function=='weight_percentile'):
            val = globals()[stat_function](realization,realization,stat_function_param1)
        else:
            if stat_function_param1 is None:
                val = globals()[stat_function](realization)
            elif stat_function_param2 is None:
                val = globals()[stat_function](realization,stat_function_param1) 
            elif stat_function_param3 is None:
                val = globals()[stat_function](realization,stat_function_param1,stat_function_param2)
            else:
                val = globals()[stat_function](realization,stat_function_param1, stat_function_param2, stat_function_param3)
        vals.append(val)
    val_mean = np.mean(vals)
    if (confidence==0):
        val_err = np.std(vals)
    else:
        dv=np.sort(np.abs(vals-val_mean))
        val_err = dv[int(confidence*len(vals))]
    return val_mean, val_err, vals

def fft_filter(x,y,kmax,kmin=0):
    x=np.array(x);y=np.array(y);
    #reorder data
    ind=x.argsort(); x=x[ind]; y=y[ind];
    #first interpolate to evenly spaced values
    mindx = np.min(x[1:]-x[:-1])
    newx = np.linspace(x[0],x[-1],num=np.int((x[-1]-x[0])/mindx))
    newy= np.interp(newx, x, y)
    #FFT, let's take care of the edges by assuming mirror symmetry at the edges
    y_fft = np.fft.fft(np.concatenate([np.flip(newy),newy,np.flip(newy)]))
    k_fft = np.fft.fftfreq(3*len(newx))/(np.max(newx)-np.min(newx))*len(newx)*np.pi
    #print np.max(k_fft)
    #Find the frequencies that we want don't want to keep
    k_ind= ((np.abs(k_fft) > kmax) | (np.abs(k_fft)< kmin) )
    #Crop the data in fourier space
    y_fft[k_ind]=0
    #Transform back
    filt_y = np.fft.ifft(y_fft)
    #interpolate to original coordinates
    filt_y_int = np.interp(x, newx, filt_y[len(newy):(2*len(newy))].real)
    return filt_y_int[ind.argsort()]

#Functions shorthands for plotting
def logfunc(log10xvals, params):
    slope=params[0]
    edgeval=params[1]
    funcval=slope*(log10xvals-edgeval)
    logout=params[2]
    if logout:
        funcval=np.log10(funcval)
    return funcval

def invdifffunc(log10xvals, params):
    vals=10**(log10xvals)
    slope=params[0]
    edgeval=10**params[1]
    funcval=slope*(1/vals-1/edgeval)
    logout=params[2]
    if logout:
        funcval=np.log10(funcval)
    return funcval

def linfunc(log10xvals, params):
    vals=10**(log10xvals)
    slope=params[0]
    edgeval=10**params[1]
    funcval=slope*(vals-edgeval)
    logout=params[2]
    if logout:
        funcval=np.log10(funcval)
    return funcval