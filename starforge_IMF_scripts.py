# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:16:13 2019

@author: David Guszejnov
"""
from __future__ import print_function
import numpy as np
from scipy import stats, optimize
import os
import GDplot #for plotting routines
import bz2
import pickle
import re #regular expressions
from GDstat import list_from_classlist as classlist
import GDstat
from matplotlib import pyplot as plt #for plotting
import matplotlib
from palettable.colorbrewer.qualitative import Set1_5, Set1_8, Dark2_8,Set3_12
import palettable
import time
from get_sink_data import sinkdata

msun=1.988e30
pc=3.08567758e16
AU=149597870700.0
tesla=1.0
mu0=4*np.pi*1e-7;
gauss=1.0e-4
yr=3.154e+7
kyr=1e3*yr
Myr=1e6*yr
Gcode=4320
G_SI = 6.674e-11
NumberDensConv=20.3813*2
gramm=1e-3; cm=1e-2
mean_limits = [1,10] #only take stars between these masses for the mean below
#99% confidence from Kroupa 2002
kroupa_hmass_slope_err=0.7
kroupa_midmass_slope_err=0.5
kroupa_lowmass_slope_err=0.7

##############################
#Completeness limit = 0.1, varying Kroupa IMF, asymmetric confidence interval centered on median
kroupa_mass_points = 10**np.linspace(1,4,num=7)
kroupa_m50 = 1.46 
kroupa_m50_err_plus = np.array([ 6.059, 17.035, 53.493, 72.744, 21.602, 15.55,  13.836])
kroupa_m50_err_minus = np.array([ 0.386, 0.366, 0.61,  0.8,  0.811, 0.811, 0.884])
kroupa_mean = 0.66
kroupa_mean_err_plus = np.array([0.788, 0.884, 1.107, 1.395, 1.151, 1.111, 1.028])
kroupa_mean_err_minus = np.array([0.203, 0.2166, 0.225, 0.24, 0.243, 0.2436, 0.248])
kroupa_median = 0.27
kroupa_median_err_plus = np.array([ 0.3 ,  0.16 , 0.128, 0.098, 0.098, 0.098, 0.098])
kroupa_median_err_minus = np.array([ 0.096, 0.072, 0.062, 0.051, 0.051, 0.051, 0.039])
kroupa_mean_lim = 2.28
kroupa_mean_lim_err_plus = np.array([4.103, 2.038, 1.17,  0.78,  0.549, 0.495, 0.463])
kroupa_mean_lim_err_minus = np.array([1.719, 0.813, 0.701, 0.522, 0.406, 0.36,  0.355])



              
def slope_to_mean_lim(slope,limits,num=50):
    x=np.linspace(limits[0],limits[1],num=num)
    return np.trapz(x**(slope+1),x=x)/np.trapz(x**(slope),x=x)

def diff_mean_lim(slope,target,limits):
    return slope_to_mean_lim(slope,limits)-target


def mean_lim_to_slope(mass,limits):
    slope_est = optimize.root(diff_mean_lim, -2.3, args=(mass,limits))['x'][0] 
    return np.clip(slope_est,-6,0)

#Estimate t_ff assuming we have a MW-like GMC with alpha=2
def t_ff(mass,R):
    G_code=4325.69
    tff = np.sqrt(3.0*np.pi/( 32*G_code*( mass/(4.0*np.pi/3.0*(R**3)) ) ) )
    return tff

def m50_corr_func(m50,alpha,mach,c_a=1.0,c_m=-3.0):
    return m50/((alpha**c_a)*(mach**c_m))

def M50_obs_fit(sfe, mass, alphaturb, sigma, cs):
    M5=mass/1e5
    sigma100=sigma/100
    cs200=cs/200
    sfe005=sfe/0.05
    m50_pred = 24 * (sfe005**0.3) * (M5**0.2) * (alphaturb**(-0.5)) * (sigma100**(-0.8)) * (cs200)
    return m50_pred

def nearest_sink_stat(filename,data_folder='sinkdata',nbin=40):
    #Get the distance of every newly formed sink to the nearest other one, estimated from snapshots
    #Load sink data
    infile=open(data_folder+'/'+filename+'.pickle','rb')
    data_orig = pickle.load(infile)
    infile.close()
    #Params
    Nsink_array=np.array([len(d.m) for d in data_orig])
    early_ind = np.argmax(Nsink_array>(Nsink_array[-1]*0.1))
    late_ind = np.argmax(Nsink_array>(Nsink_array[-1]*0.9))
    dataranges=[np.arange(0,early_ind), np.arange(late_ind,len(data_orig)), np.arange(0,len(data_orig)), np.arange(0,len(data_orig)), np.arange(0,len(data_orig)), np.arange(0,len(data_orig)) ]
    labels=['First 10\% of sinks', 'Last 10\% of sinks', '$\mathrm{M}_\mathrm{sink}>10\,\mathrm{M_{\odot}}$',\
            '$\mathrm{M}_\mathrm{sink}<0.1\,\mathrm{M_{\odot}}$','Age $<$ 2kyr','All sinks']
    masslimit=[None,None,'high','low','young',None]
    lowmass=0.1
    highmass=10.0
    min_t=2e3*yr/(pc/1)
    light_sinks=data_orig[-1].id[data_orig[-1].m<lowmass]
    massive_sinks=data_orig[-1].id[data_orig[-1].m>highmass]
    
    for datarange, label,lim in zip(dataranges,labels,masslimit):
        data= np.array(data_orig)[datarange]
        #Find new sinks every snapshot and get their distance to the nearest sinks
        distances=[]
        for i,d in enumerate(data[:]):
            if (len(d.m)>1):
                #get new sinks
                if (i==0):
                    if (datarange[0]==0):
                        newids=d.id
                    else:
                        oldids=data_orig[datarange[0]-1].id
                        newids=np.setdiff1d(d.id,oldids)
                else:
                    oldids=data[i-1].id
                    newids=np.setdiff1d(d.id,oldids)
                if (lim=='high'):
                    newids=np.intersect1d(massive_sinks,newids)
                if (lim=='low'):
                    newids=np.intersect1d(light_sinks,newids)
                if (lim=='young'):
                    newids=np.intersect1d(d.id[(d.t-d.formation_time)<min_t],newids)
                #Go over new sinks
                for j in range(len(newids)):
                    x0=d.x[d.id==newids[j]]
                    dx=d.x-x0
                    dr=np.sqrt(np.sum(dx**2,axis=1))
                    distances.append(np.min(dr[dr>0]))
        #convert to AU
        distances=np.array(distances)*pc/AU/20
        #Now let's make a histogram of the distances
        hist,edges =np.histogram(np.log10(distances), bins=nbin, density=False)
        bins=GDstat.edge_to_bin(edges)
        plt.step(bins,np.log10(hist),label=label,alpha=0.9)
    plt.xlabel(r'Log d [dx]')
    plt.ylabel('Log Number of sinks')
    plt.axvline(x=0,color='k',linestyle=':')
    plt.xlim(plt.xlim()[0],plt.xlim()[1]+0.5)
    plt.legend()
    filename=filename+'_nearest_sink_hist'
    plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' ); plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
    plt.show()
 
def energy_to_temp(internalenergy,metallicity, electronabundance,density ):
    global const
    #Get H2 dissociation temperature
    Tmol=100+np.zeros(len(internalenergy)) #temperature above which H2 breaks up
    Tmol[density>0] *= (density[density>0]*const.NumberDensConv) / 100.0 #based on Glover Clark 2012, lifted from cooling.c
    Tmol[Tmol>8000.0] = 8000.0; #max temperature for Tmol
    #Get mass fractions
    if (len(metallicity.shape)==2 ): 
        helium_mass_fraction=metallicity[:,1]
        metal_mass_fraction=0*metallicity[:,0]
    else:
        helium_mass_fraction=0.24+np.zeros(len(internalenergy)) #default mass fraction of He in GIZMO
        metal_mass_fraction=np.zeros(len(internalenergy))
    fmol=np.zeros(len(internalenergy)) #assume no H2 to start
    Htotfrac=1.0-helium_mass_fraction-metal_mass_fraction #all H and H2, ignore metals, small correction
    for iter in range(10): #10 iterations to converge
        #get mean molecular weight
        mu=1.0/( Htotfrac*(1-0.5*fmol) + helium_mass_fraction/4.0 + Htotfrac*electronabundance + metal_mass_fraction/(16.0+12.0*fmol) ) #from GIZMO get_mu routine in cooling.c
        #get temperature
        temp=internalenergy*(const.gamma-1.0)*(const.mp/const.kb)*mu
        #recalculate molecular fraction
        fmol=1.0/(1.0+ (temp/Tmol)**2 )
    return temp 
 
def default_GMC_R(mass):
    if mass==2e2: R=1;
    elif mass==2e3: R=3;
    elif mass==2e4: R=10;
    elif mass==2e5: R=30;
    elif mass==2e6: R=100;
    else: R=((mass/2e4)**0.5)*10
    return R
 
def convert_leading_zero_string_to_float(string):
    fact=1
    if string[0]=='0':
        fact=10**(1-len(string))    
    return (float(string)*fact)

    
def pick_data_at_SFE(filenames, target_SFE=0.1,data_folder='sinkdata',use_tff_instead_of_SFE=False,target_tff=None, verbose=True):
    if use_tff_instead_of_SFE and (target_tff is None):
        target_tff=target_SFE
    #First we should load all the files
    datalist=[]; initmass_list=[]; res_list=[]; sfe_evol_list=[]; tff_evol_list=[]; max_sfe_list=[];R_list=[];cs_list=[];alpha_list=[]
    for f in filenames:
        if verbose: print("Processing "+f)
        infile=open(data_folder+'/'+f+'.pickle','rb')
        data = pickle.load(infile)
        datalist.append(data)
        infile.close()
        #Lets get the initial gas mass for each, which we can only get from the name
        try:
            initmass=float(re.search('M\de\d', f).group(0).replace('M',''))
        except:
            print("Initial cloud mass can not be determined from filename", f)
            R_guess = np.mean(data[-1].x)*0.2
            initmass = 2e4*(R_guess/10)**2;
            initmass = float('%.1g'%(10**np.round(np.log10(initmass),decimals=1)) )
            print("Estimated initial mass is %g, assuming box size=10R and a MW-like mass-size relation"%(initmass))
        initmass_list.append(initmass)
        if re.search('R\d', f) is None:
            R=default_GMC_R(initmass)
        else:
            R=float(re.search('R\d\d*', f).group(0).replace('R',''))
        R_list.append(R)
        if re.search('alpha\d', f) is None:
            alpha=2.0
        else:
            alpha=convert_leading_zero_string_to_float(re.search('alpha\d*', f).group(0).replace('alpha',''))
        alpha_list.append(alpha)
        if f == 'M2e4_C_M_J_vhiT_2e7':
            Tfloor=60.
        elif f == 'M2e4_C_M_J_hiT_2e7':
            Tfloor=30.
        elif f == 'M2e4_C_M_J_hiT_fine_2e7':
            Tfloor=30.
        else:
            Tfloor=10.
        cs_list.append(np.sqrt(Tfloor/10)*200)
        if 'Res' in f:
            npar=float(re.search('Res\d*', f).group(0).replace('Res',''))**3
        else:
            try:
                npar=float(re.search('_\de\d', f).group(0).replace('_',''))
            except:
                print("Resolution not set in filename, assuming fiducial 1e-3 Msun")
                npar = initmass/0.001
        res=initmass/npar
        res_list.append(res)
        tff = t_ff(initmass,R)
        tff_evol_list.append(np.array([d.t/tff for d in data]))
        #Now we have an estimate for the star formation efficiency
        sfe_evol = np.array([np.sum(d.m)/initmass for d in data])
        sfe_evol_list.append(sfe_evol)
        max_sfe_list.append(np.max(sfe_evol))  
    #Let's decide if we want to pick at fixed SFE or tff
    if use_tff_instead_of_SFE:
        target_metric = target_tff
        compare_metric = tff_evol_list
    else:
        target_metric = target_SFE
        compare_metric = sfe_evol_list
    if target_metric is None:
        comparison_ind = [-1 for metric_evol in compare_metric]
        comparison_point = None
    elif target_metric=='Max':
        #Find the index for highest sfe
        comparison_ind = [np.argmax(metric_evol) for metric_evol in compare_metric]
        comparison_point = None
    elif hasattr(target_metric, "__iter__"):
        #Find the index for each closest to the value described by the target_metric array
        comparison_ind = [np.argmax(metric_evol>=target) for target,metric_evol in zip(target_metric,compare_metric)]
        comparison_point = None
    else:
        #Let's find what SFE we will compare at
        comparison_point= np.min([np.min([np.max(metric_evol) for metric_evol in compare_metric ]),target_metric])
        if comparison_point<target_metric:
            print("Not all models reach the target matric of %g, using %g instead"%(target_metric,comparison_point))
            print("Lowest model is "+filenames[np.argmin(max_sfe_list)]+" at metric of %g"%(np.min([np.max(metric_evol) for metric_evol in compare_metric ])))
        #Corresponding indices, the first ones with at least that SFE
        comparison_ind = [np.argmax(sfe_evol>=comparison_point) for sfe_evol in compare_metric]
    #Let's make the data lists, for each model we need the sink data at the snapshot where its SFE is closest to the target
    result = [datalist[i][comparison_ind[i]] for i in range(len(datalist))]
    return result, comparison_point, initmass_list, comparison_ind, datalist, res_list, R_list, cs_list,alpha_list


   
class sink_mass_stats:
    def __init__(self,data,initmass,label,res,R,cs,alpha,normalize_masses=False,calculate_errors=True,\
                 M50_error_only=True,confidence=0.95,main_ind=0,min_particle_num=1):
        self.label = label
        self.minmass=0
        if min_particle_num>1:
            self.minmass = res*min_particle_num
        if normalize_masses:
            for d in data:
                d.m /= initmass
        # self.minmass = 0.1 #need to hardcode for resolution tests
        # print("Enforcing minimum mass of %g"%(self.minmass))
        minmass=self.minmass
        self.data = data
        self.t_orig = np.array(classlist(self.data,"t"))
        #normalize time
        self.t=self.t_orig-self.t_orig[0]
        self.snapnum = np.array(classlist(self.data,"snapnum"))
        self.Nsink = np.array([len(d.m) for d in self.data])
        self.Msink = np.array([np.sum(d.m) for d in self.data])
        self.initmass = initmass
        self.R = R
        self.cs=cs
        self.alpha=alpha
        if normalize_masses:
            self.sfe = self.Msink
            self.res = res/self.initmass
        else:
            self.sfe = self.Msink/self.initmass
            self.res = res

        #Basic stats
        self.mass_50 = np.array([GDstat.weight_median(d.m[d.m>=minmass],d.m[d.m>=minmass]) for d in self.data])
        self.mass_10 = np.array([GDstat.weight_percentile(d.m[d.m>=minmass],d.m[d.m>=minmass],0.1) for d in self.data])
        self.mass_25 = np.array([GDstat.weight_percentile(d.m[d.m>=minmass],d.m[d.m>=minmass],0.25) for d in self.data])
        self.mass_75 = np.array([GDstat.weight_percentile(d.m[d.m>=minmass],d.m[d.m>=minmass],0.7) for d in self.data])
        self.mass_90 = np.array([GDstat.weight_percentile(d.m[d.m>=minmass],d.m[d.m>=minmass],0.9) for d in self.data])
        self.mass_100 = np.array([np.max(d.m,initial=0) for d in self.data])
        self.mass_med = np.array([np.median(d.m[d.m>=minmass]) for d in self.data])
        self.mass_mean = np.array([np.mean(d.m[d.m>=minmass]) for d in self.data])
        self.mass_mean_lim = np.array([GDstat.weight_mean_within_range(d.m[d.m>=minmass],0,mean_limits) for d in self.data])
        self.lim_slope = np.array([mean_lim_to_slope(m,mean_limits) for m in self.mass_mean_lim])
        #Errors of basic stats
        self.mass_mean_err = np.zeros(len(data))
        self.mass_med_err = np.zeros(len(data))
        self.mass_10_err = np.zeros(len(data))
        self.mass_25_err = np.zeros(len(data))
        self.mass_50_err = np.zeros(len(data))
        self.mass_75_err = np.zeros(len(data))
        self.mass_90_err = np.zeros(len(data))
        self.mass_mean_lim_err = np.zeros(len(data))
        self.lim_slope_err =  np.zeros(len(data))
        if calculate_errors:
            if calculate_errors is True:
                N_MC = 20
            else:
                N_MC = calculate_errors
            if (main_ind==0):
                print("Starting error calculation for "+label+" using %d MC realizations for each of the %d snapshots"%(N_MC, len(self.data)))
                for i,d in enumerate(self.data):
                    if len(d.m[d.m>=minmass])>1:
                        _, self.mass_50_err[i], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.5, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                        _, self.mass_mean_err[i], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='npmean',N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                        _, self.mass_med_err[i], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='npmedian',N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                        _, self.mass_mean_lim_err[i], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_mean_within_range',stat_function_param1=0, stat_function_param2=mean_limits, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                        if not M50_error_only:
                            _, self.mass_10_err[i], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.1, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                            _, self.mass_25_err[i], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.25, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                            _, self.mass_75_err[i], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.75, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                            _, self.mass_90_err[i], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.9, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
            else:  
                d=self.data[main_ind]
                if len(d.m[d.m>=minmass])>1:
                    _, self.mass_50_err[main_ind], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.5, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                    _, self.mass_mean_err[main_ind], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='npmean',N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                    _, self.mass_med_err[main_ind], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='npmedian',N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                    _, self.mass_mean_lim_err[main_ind], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_mean_within_range',stat_function_param1=0, stat_function_param2=mean_limits, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                    if not M50_error_only:
                        _, self.mass_10_err[main_ind], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.1, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                        _, self.mass_25_err[main_ind], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.25, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                        _, self.mass_75_err[main_ind], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.75, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
                        _, self.mass_90_err[main_ind], _ = GDstat.MC_stat_func_guess(d.m[d.m>=minmass], stat_function='weight_percentile',stat_function_param1=0.9, N_MC=N_MC,fixed_sum=d.m[d.m>=minmass],confidence=confidence)
            self.lim_slope_err= np.array([0.5*(mean_lim_to_slope(m+dm,mean_limits)-mean_lim_to_slope(m-dm,mean_limits)) for m,dm in zip(self.mass_mean_lim,self.mass_mean_lim_err)])
                    
        
        
    #Statistics
    def PDFs(self,logbin_edges,no_zero_bins=False,min_particle_num=1):
        #Init
        self.mass_logbin_edges = logbin_edges
        self.mass_logbins = GDstat.edge_to_bin(self.mass_logbin_edges)
        self.IMF=[]; self.mass_PDF=[]; 
        minmass = self.res * min_particle_num
        for d in self.data:
            IMF, _ =np.histogram(np.log10(d.m[d.m>=minmass]), bins=logbin_edges, density=False)
            mass_PDF, _ =np.histogram(np.log10(d.m[d.m>=minmass]), bins=logbin_edges,\
                                                           density=False,weights=d.m[d.m>=minmass])
            #Correction to avoid zeros in log plot
            if no_zero_bins:
                IMF = np.float64(IMF)+1e-5
                mass_PDF += np.float64(mass_PDF)+1e-5*np.min(mass_PDF[mass_PDF>0])
            #store
            self.IMF.append(IMF); self.mass_PDF.append(mass_PDF);
    def CDFs(self,min_particle_num=1):
        #Init
        self.CDF_mass=[]; self.CDF=[];#self.logCDF=[]; 
        self.inv_CDF_mass=[]; self.inv_CDF=[];#self.loginv_CDF=[]
        self.CMF_mass=[]; self.CMF=[];#self.logCMF=[]; 
        self.inv_CMF_mass=[]; self.inv_CMF=[];#self.loginv_CMF=[]
        minmass = self.res * min_particle_num
        #print(self.label, self.res, self.initmass,min_particle_num )
        for d in self.data:
            CDF_mass, CDF = GDstat.make_CDF(np.log10(d.m[d.m>=minmass]),start_from_high=True,normed=False)
            #logCDF = np.log10(CDF)
            inv_CDF_mass, inv_CDF = GDstat.make_CDF(np.log10(d.m[d.m>=minmass]),start_from_high=False,normed=False)
            #loginv_CDF = np.log10(inv_CDF)
            CMF_mass, CMF = GDstat.make_CDF(np.log10(d.m[d.m>=minmass]),start_from_high=True,normed=False,\
                                                      weights=d.m[d.m>=minmass])
            #logCMF = np.log10(CMF)
            inv_CMF_mass, inv_CMF = GDstat.make_CDF(np.log10(d.m[d.m>=minmass]),start_from_high=False,\
                                                              normed=False, weights=d.m[d.m>=minmass])
            #loginv_CMF = np.log10(inv_CMF)
            #Store
            self.CDF_mass.append(CDF_mass); self.CDF.append(CDF);# self.logCDF.append(logCDF)
            self.inv_CDF_mass.append(inv_CDF_mass); self.inv_CDF.append(inv_CDF);# self.loginv_CDF.append(loginv_CDF)
            self.CMF_mass.append(CMF_mass); self.CMF.append(CMF);# self.logCMF.append(logCMF)
            self.inv_CMF_mass.append(inv_CMF_mass); self.inv_CMF.append(inv_CMF);# self.loginv_CMF.append(loginv_CMF)
    def caculate_stats(self,logbin_edges,no_zero_bins=False,min_particle_num=1):
        self.PDFs(logbin_edges,no_zero_bins=no_zero_bins)
        self.CDFs()  
    def find_index(self,target_time=None, target_SFE=None):
        if target_time is None:
            if target_SFE is None:
                print("One of target_time or target_SFE needs to be specified")
                return -1
            else:
                #pick out the index corresponding to the target SFE
                return np.argmax(self.sfe>=target_SFE) 
        else:
            #pick out the index corresponding to the target time
            return np.argmax(self.t>=target_time)

def set_colors_and_styles(colors, styles, N, multiples=1, dark=False, cmap=None,sequential=False):
    if colors is None:
        if sequential and (N<=4):
            #colors_base = ['#a1dab4', '#41b6c4', '#2c7fb8','#253494']
            colors_base = [(161/255,218/255,180/255),(65/255,182/255,196/255),(44/255,127/255,184/255),(37/255,52/255,148/255) ]
            colors = [colors_base[int(i/N*len(colors_base))] for i in range(N)]
        else:
            if (N<=5):
                #colors=['k', 'g', 'r', 'b', 'm', 'brown', 'orange', 'cyan', 'gray', 'olive']
    #            if dark:
    #                colors=Dark1_5.mpl_colors[:N]
    #            else:
                    colors=Set1_5.mpl_colors[:N]
            elif (N<=8):
                if dark:
                    colors=Dark2_8.mpl_colors[:N]
                else:
                    colors=Set1_8.mpl_colors[:N]
            elif (N<=12):
                    colors=Set3_12.mpl_colors[:N] 
            else:
                if cmap is None:
                    cm=matplotlib.cm.get_cmap()
                else:
                    cm=matplotlib.cm.get_cmap(cmap)
                colors=[cm(i/N) for i in range(N)]
    if styles is None:
        styles=np.full(N,'-')
    if multiples>1:
        colors_old=colors[:]; styles_old=styles[:]
        colors=[]; styles=[]
        for i in range(N):
            for j in range(multiples):
                colors.append(colors_old[i])
                styles.append(styles_old[j])
    return colors, styles


def kroupa_IMF(logmass,offset=0, log_units=True, hmass_slope=-2.3, midmass_slope=-1.3, lowmass_slope=-0.3):
    #Kroupa 2002 IMF
    m1=0.008; m2=0.08; m3=0.5
    #Find normalization
    C1 = 4.48693; C2 = C1*m2**(lowmass_slope-midmass_slope); C3 = C2*m3**(midmass_slope-hmass_slope)
    if log_units:
        ds=1
    else:
        ds=0
    mass=10**np.array(logmass)
    if hasattr(mass, "__iter__"):
        imf=np.zeros_like(mass)
        ind1=(mass>=m1)&(mass<m2)
        imf[ind1]= C1 * (mass[ind1]**(lowmass_slope+ds))
        ind2=(mass>=m2)&(mass<m3)
        imf[ind2]= C2 * (mass[ind2]**(midmass_slope+ds))
        ind3=(mass>=m3)
        imf[ind3]= C3 * (mass[ind3]**(hmass_slope+ds))
        return imf*(10**offset)
    else:
        if ( (mass>=m1) and (mass<m2) ):
            return C1 * (mass**(lowmass_slope+ds))
        elif ( (mass>=m2) and (mass<m3) ):
            return C2 * (mass**(midmass_slope+ds))
        elif (mass>=m3):
            return C3 * (mass**(hmass_slope+ds))
        else:
            return 0
        


def chabrier_IMF(logmass,offset=0, log_units=True, systemIMF=True):
    #Chabrier 2005 IMF
    imf=0*logmass
    if systemIMF:
        logmc=np.log10(0.25); logm1=np.log10(1); sigma=0.55; c1=0.076; c2=0.041
    else:
        logmc=np.log10(0.2); logm1=np.log10(1); sigma=0.55; c1=0.093; c2=0.041
    if log_units:
        ds=1
    else:
        ds=0
    ind1=(logmass>=logm1)
    imf[ind1] = c2 * ((10**logmass[ind1])**(-2.35+ds))
    ind2=(logmass<logm1)
    imf[ind2] = c1 * np.exp( -0.5*((logmass[ind2] - logmc)**2.0)/(sigma**2) )
    if not log_units:
        imf[ind2] += logmass[ind2]
    return imf*(10**offset)


def mass_stat_evolplots(stat,colors=None, styles=None):
    mass_stats = [np.log10(stat.mass_10),np.log10(stat.mass_25),np.log10(stat.mass_50),np.log10(stat.mass_75),np.log10(stat.mass_90)]
    sfe_vals = [np.log10(stat.sfe) for i in range(len(mass_stats))]
    labels = ["10", "25","50","75","90"]
    colors, styles = set_colors_and_styles(colors, styles, len(mass_stats))
    #Do the plot
    GDplot.plot(sfe_vals,mass_stats,'','log SFE','Log M]',stat.label+'_mass_stat_evol',\
                styles=styles, labels=labels, colors=colors, overtext=stat.label, overtextfontsize=20, overtextcoord=[0.01,0.9])

    
    
def evolplot(x, y,statlist,labels, xlabel, ylabel, filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=False, kroupa_to_use=None,\
             overtextfontsize=13, plot_fitlines=False,labelspacing=0,legendfontsize=14,plot_obs_line_only=False,fitline_label=None,\
             noplot=False,yerr=None,xmax=None,legendloc='best',use_sampling_error=True,smooth_scale=None):
    if not (smooth_scale is None): #get same number of points within plotted region
        for i,yvals in enumerate(y): #smoothing
            window = (int(smooth_scale/np.min(np.diff(x[i])[np.diff(x[i])>0]))//2)*2+1
            if window>=3:
                y_avg, valid_ind1, valid_ind2 = GDstat.moving_avg(yvals,N=window)
                y[i] = y_avg; x[i] = x[i][valid_ind1:valid_ind2]
    #Plot the evolution of a chosen mass scale of the sink particles in the simulation
    GDplot.plot(x,y,'',xlabel,ylabel,filename,labels=labels,colors=colors,styles=styles,noplot=True,savefig=False)
    if not (overtext is None):
        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left',\
                 verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
    if plot_obs_IMF and not (kroupa_to_use is None):
        #Interpolate error values
        if kroupa_to_use=="m50":
            kroupa_val = np.log10(kroupa_m50); 
            kroupa_errval_orig_plus = np.log10(1+kroupa_m50_err_plus/kroupa_m50); kroupa_errval_orig_minus = -np.log10(1-kroupa_m50_err_minus/kroupa_m50);
        elif kroupa_to_use=="median":
            kroupa_val =  np.log10(kroupa_median); 
            kroupa_errval_orig_plus = np.log10(1+kroupa_median_err_plus/kroupa_median); kroupa_errval_orig_minus = -np.log10(1-kroupa_median_err_minus/kroupa_median);
        elif kroupa_to_use=="mean":
            kroupa_val =  np.log10(kroupa_mean); 
            kroupa_errval_orig_plus = np.log10(1+kroupa_mean_err_plus/kroupa_mean); kroupa_errval_orig_minus = -np.log10(1-kroupa_mean_err_minus/kroupa_mean);
        elif kroupa_to_use=="mean_lim":
            kroupa_val =  np.log10(kroupa_mean_lim);
            kroupa_errval_orig_plus = np.log10(1+kroupa_mean_lim_err_plus/kroupa_mean_lim); kroupa_errval_orig_minus = -np.log10(1-kroupa_mean_lim_err_minus/kroupa_mean_lim);
        elif kroupa_to_use=="slope":
            kroupa_val = -2.3;  plot_obs_line_only=True;
            kroupa_errval_orig_plus = kroupa_hmass_slope_err*2/3 * np.ones_like(kroupa_mean_err_plus); kroupa_errval_orig_minus=kroupa_errval_orig_plus
        if plot_obs_line_only:
            plt.axhline(y=kroupa_val,color='k',linestyle='--',label='Kroupa (2002)')
            kroupa_errval_plus = kroupa_errval_orig_plus[-1]; kroupa_errval_minus = kroupa_errval_orig_minus[-1];
            plt.fill_between(plt.xlim(),kroupa_val-kroupa_errval_minus,kroupa_val+kroupa_errval_plus,alpha=0.3,color='k')
        else:
        #get SFE
            sfe = [np.log10(s.sfe[s.sfe>0]) for s in statlist]
            xvals = np.linspace(np.min(np.concatenate(sfe)),np.max(np.concatenate(sfe)),num=100)
            kroupa_val_error_plus = np.interp(xvals,np.log10(kroupa_mass_points/np.min([s.initmass for s in statlist])),kroupa_errval_orig_plus)
            kroupa_val_error_minus = np.interp(xvals,np.log10(kroupa_mass_points/np.min([s.initmass for s in statlist])),kroupa_errval_orig_minus)
            plt.axhline(y=kroupa_val,color='k',linestyle='--',label='Kroupa (2002)')
            plt.fill_between(xvals,kroupa_val-kroupa_val_error_minus,kroupa_val+kroupa_val_error_plus,alpha=0.3,color='k')
        plt.legend()
    if (use_sampling_error) and (not (yerr is None)):
        if not hasattr(yerr[0], "__iter__"): yerr = [yerr]
        for i,yerr_vals in enumerate(yerr):
            plt.fill_between(x[i],y[i]-yerr_vals,y[i]+yerr_vals,alpha=0.6/len(x),color=colors[i],zorder=10)
    if plot_fitlines:
        if plot_fitlines is True: 
            slope = 0.333
        else:
            slope = plot_fitlines
        fitx = 0.7*(plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
        fitpoint = [fitx,np.interp(fitx, x[0], y[0])]
        fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle=':')]
        GDplot.add_line_to_plot(plt.gca(),slope,point_to_include=fitpoint,num=100,style=':',color='k')       
    plt.xlim([np.min(np.concatenate(x)) ,np.max(np.concatenate(x))])
    plt.ylim(plt.ylim()[0],plt.ylim()[1]+0.6)
    if plot_fitlines and (legendloc=='best'): legendloc=2
    leg=plt.legend(loc=legendloc, fontsize=legendfontsize,labelspacing=labelspacing);
    if plot_fitlines:
        if fitline_label is None: fitline_label=r'$M_{\mathrm{50}}\propto\mathrm{SFE}^{1/3}$'
        plt.legend(fitlines,[fitline_label],loc=4, fontsize=legendfontsize,labelspacing=labelspacing)
    plt.gca().add_artist(leg)
    if not (xmax is None):
        xlim = plt.xlim()
        if (xlim[1]>xmax): plt.xlim([xlim[0],xmax])
    GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
    plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' ); #plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
    if not noplot:
        plt.show()

def model_comparison(filenames,target_SFE='Max',modelnames=None,label='',data_folder='output',logmassbin_edges=None,\
                     colors=None, styles=None,overtext=None, overtextcoord=[0.02,0.95],overtextfontsize=18, xlim=None,\
                     CDF_ylim=None, plot_obs_IMF=False,plot_fitlines=False, fitpos=None, do_mass_stat_evolplots=False, do_CDFs=False,do_inv_CDFs=False,kroupa_norm_point_logM = 0,\
                     normalize_masses=False,do_step_plots=False,calculate_errors=False, min_particle_num=1,legendfontsize=14,\
                     addM50_to_plots=False, compare_m50_groups=None, M50_vs_M_at_SFE=None,labelspacing=0,\
                     put_symbol=True,error_only_for_SFE_snap=True,shade_low_masses=True,imf_max_normalize=False, extra_vlines=None, plot_mass_IMF=False,\
                     plot_young_IMF=False, use_tff_instead_of_SFE=False, target_tff=None,maxtff=None):
    nbins_default=20
    if hasattr(target_SFE, "__iter__") and target_SFE!='Max': 
        plot_only_IMF = True
    else:
        plot_only_IMF = False
    if modelnames is None:
        modelnames=filenames
    if (fitpos is None):
        fitpos=[ [0.72,0], [0.9,0.2], [0.7,0.2], [0.75,1.45], [0.5,0.5] ]
    colors, styles = set_colors_and_styles(colors, styles, len(filenames))
    labels=modelnames
    
    if do_step_plots:
        plotfunc=GDplot.step_plot
    else:
        plotfunc=GDplot.plot
     
    print("Processing models",modelnames)
    #Let's find the snapshots at te target SFE
    data_at_SFE, comparison_SFE, initmass_list, snap_ind, datalist, res_list, R_list, cs_list,alpha_list = pick_data_at_SFE(filenames, target_SFE=target_SFE,data_folder=data_folder, use_tff_instead_of_SFE=use_tff_instead_of_SFE, target_tff=target_tff)
    
    #We need to set the bin forthe PDFss if they are not prescribed
    if logmassbin_edges is None:
        if xlim is not None:
            if np.log10(xlim[1]/xlim[0])>5:
                nbins=int(np.log10(xlim[1]/xlim[0])*(nbins_default/5))
            else:
                nbins=nbins_default
            logmassbin_edges = np.linspace(xlim[0],xlim[1],num=nbins+1)
        else:
            #we need to get the range of masses, so let's find teh lowest and highest in all models
            if not normalize_masses:
                minmass = np.min([np.min(d.m) for d in data_at_SFE])
                maxmass = np.max([np.max(d.m) for d in data_at_SFE])
            else:
                minmass = np.min([np.min(d.m/initmass) for d,initmass in zip(data_at_SFE,initmass_list)])
                maxmass = np.max([np.max(d.m/initmass) for d,initmass in zip(data_at_SFE,initmass_list)])
            if (min_particle_num>=3):
                minmass *= min_particle_num/3.0
            logmassbin_edges = np.linspace(np.log10(minmass),np.log10(maxmass),num=nbins_default+1)
    logmassbin_sizes = logmassbin_edges[1:]-logmassbin_edges[:-1]
 
    #List of classes that have the statistics we want
    if error_only_for_SFE_snap and ( (calculate_errors is False) or (calculate_errors==0) ):
        choose_factor=1
    else:
        choose_factor=0
    statlist = [sink_mass_stats(datalist[i],initmass_list[i],modelnames[i],res_list[i],R_list[i],cs_list[i],alpha_list[i],normalize_masses=normalize_masses,calculate_errors=calculate_errors,main_ind=(snap_ind[i]*choose_factor),min_particle_num=min_particle_num) for i in range(len(data_at_SFE))]
    
    #Let's make the PDFs, CDFs and mass PDFs
    for s,ind in zip(statlist,snap_ind):
        s.caculate_stats(logmassbin_edges,no_zero_bins=do_step_plots,min_particle_num=min_particle_num)
        if normalize_masses:
            msun1 = 1/s.initmass
        else:
            msun1 = 1.0
        #print out the mass statistics
        if calculate_errors:
            print(s.label+" mass percentiles 10: %g+-%g    50 %g+-%g    90: %g+-%g"%(s.mass_10[ind],s.mass_10_err[ind],s.mass_50[ind],s.mass_50_err[ind],s.mass_90[ind],s.mass_90_err[ind]))
            print(s.label)
            print(s.mass_50[ind])
            print(s.mass_50_err[ind])
        else:
            print(s.label+' using snapindex %d'%(ind))
            print("mass percentiles 10: %g    50 %g    90: %g"%(s.mass_10[ind],s.mass_50[ind],s.mass_90[ind]))
        print('Max SFE is %g'%np.max(s.sfe))
        sfemin = 1e-2
        if np.max(s.sfe)>sfemin:
            for mass_scale, mass_name in zip([s.mass_50, s.mass_mean, s.mass_med], ['M50', 'mean mass', 'median mass']):
                p, p_err, _ = GDstat.polyfit_with_error(np.log10(s.sfe[s.sfe>sfemin]),\
                           np.log10(mass_scale[s.sfe>sfemin]),\
                           1, yerr=None,verbose=False)
                print("SFE exponent for %s is roughly %g +- %g"%(mass_name,p[0],p_err[0]))
            print("Number of >1 Msun sinks: %d \t Total number of sinks %d"%(np.sum(s.data[ind].m>msun1),s.Nsink[ind]))
            print("Number of >10 Msun sinks: %d \t Total number of sinks %d"%(np.sum(s.data[ind].m>msun1*10),s.Nsink[ind]))
    if do_mass_stat_evolplots:
        for s in statlist:
            mass_stat_evolplots(s)
            
    if normalize_masses:
        mass_label='Log $\mathrm{M/M_0}$'
        m50_label='Log $\mathrm{M_{50}/M_0}$'
        cum_mass_label='Log $\mathrm{M_{tot}}(>M)/\mathrm{M_0}$'
        mass_imf_label='Log $dM_{\mathrm{tot}}/dlogM$'
    else:
        mass_label='Log $\mathrm{M_{sink}}$ [$\mathrm{M_{\odot}}$]'
        cum_mass_label='Log $\mathrm{M_{tot}}$(>M)'  
        m50_label='Log $\mathrm{M_{50}}$ [$\mathrm{M_{\odot}}$]'
        mass_imf_label='Log $dM_{\mathrm{tot}}/dlogM$ [$\mathrm{M_{\odot}}$]'
    imf_label_default='Log d$\mathrm{N_{sink}}$/dlog$\mathrm{M_{sink}}$'
    imf_label_shifted=imf_label_default+' [shifted]'
    #Now we are ready for plotting
    if not (plot_only_IMF):
        logsfe_min=-4.0; sfe_min = 10**logsfe_min
        sfe = [np.log10(s.sfe[s.sfe>=(10**logsfe_min)]) for s in statlist]
        #Estimate t_ff
        t_norm_est = [s.t[s.sfe>sfe_min]/t_ff(s.initmass,s.R) for s in statlist]
        
        #Mass weighted median mass vs SFE or tff
        m50 = [np.log10(s.mass_50[s.sfe>=(10**logsfe_min)]) for s in statlist]
        if calculate_errors:
            yerr = [np.log10(1.0+s.mass_50_err[s.sfe>=(10**logsfe_min)]/s.mass_50[s.sfe>=(10**logsfe_min)]) for s in statlist]
        else:
            yerr = None
        filename='m50_SFE'+label
        
        evolplot(sfe, m50,statlist,labels, 'Log SFE', m50_label, filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, kroupa_to_use="m50", overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize)
        filename='m50_tnorm'+label
        evolplot(t_norm_est, m50,statlist,labels, r't/$\mathrm{t_{ff}}$', m50_label, filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, kroupa_to_use="m50", overtextfontsize=overtextfontsize, plot_fitlines=False,labelspacing=labelspacing,legendfontsize=legendfontsize,plot_obs_line_only=True, yerr=yerr, xmax=maxtff)
        #Median mass vs SFE or tff
        mmedian = [np.log10(s.mass_med[s.sfe>=(10**logsfe_min)]) for s in statlist]
        if calculate_errors:
            yerr = [np.log10(1.0+s.mass_med_err[s.sfe>=(10**logsfe_min)]/s.mass_med[s.sfe>=(10**logsfe_min)]) for s in statlist]
        else:
            yerr = None
        filename='Mmed_SFE'+label
        evolplot(sfe, mmedian,statlist,labels, 'Log SFE', 'Log $M_\mathrm{med}$ [$\mathrm{M_\odot}$]', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, kroupa_to_use="median", overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize)
        filename='Mmed_tnorm'+label
        evolplot(t_norm_est, mmedian,statlist,labels, r't/$\mathrm{t_{ff}}$', 'Log $M_\mathrm{med}$ [$\mathrm{M_\odot}$]', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, kroupa_to_use="median", overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize,plot_obs_line_only=True, yerr=yerr, xmax=maxtff)
        
        #Mean mass vs SFE or tff
        mmean = [np.log10(s.mass_mean[s.sfe>=(10**logsfe_min)]) for s in statlist]
        if calculate_errors:
            yerr = [np.log10(1.0+s.mass_mean_err[s.sfe>=(10**logsfe_min)]/s.mass_mean[s.sfe>=(10**logsfe_min)]) for s in statlist]
        else:
            yerr = None
        filename='Mmean_SFE'+label
        evolplot(sfe, mmean,statlist,labels, 'Log SFE', 'Log $M_\mathrm{mean}$ [$\mathrm{M_\odot}$]', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, kroupa_to_use="mean", overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize)
        filename='Mmean_tnorm'+label
        evolplot(t_norm_est, mmean,statlist,labels, r't/$\mathrm{t_{ff}}$', 'Log $M_\mathrm{mean}$ [$\mathrm{M_\odot}$]', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, kroupa_to_use="mean", overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize,plot_obs_line_only=True, yerr=yerr, xmax=maxtff)
        
        #Slope in limited range vs SFE
        min_slope=-5
        sfe = [np.log10(s.sfe[s.lim_slope>min_slope]) for s in statlist]
        t_norm_est = [s.t[s.lim_slope>min_slope]/t_ff(s.initmass,s.R) for s in statlist]
        lim_slope = [s.lim_slope[s.lim_slope>min_slope] for s in statlist]
        if calculate_errors:
            yerr = [s.lim_slope_err[s.lim_slope>min_slope] for s in statlist]
        else:
            yerr = None
        filename='slope_SFE'+label
        evolplot(sfe, lim_slope,statlist,labels, 'Log SFE', 'Slope between $1\,-\,10\,\mathrm{M_\odot}$', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, kroupa_to_use="slope", overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize,plot_obs_line_only=True, yerr=yerr)
        filename='slope_tnorm'+label
        evolplot(t_norm_est, lim_slope,statlist,labels, r't/$\mathrm{t_{ff}}$', 'Slope between $1\,-\,10\,\mathrm{M_\odot}$', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, kroupa_to_use="slope", overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize,plot_obs_line_only=True, yerr=yerr, xmax=maxtff,smooth_scale=0.01)

        #Mdot vs t/tff        
        filename='epsff_tff'+label
        sfe_min=1e-6; logsfe_min=np.log10(sfe_min);
        pre_sim_time = [s.t_orig[0]-s.snapnum[0]*np.median(np.diff(s.t_orig)) for s in statlist]
        tnorm_alt = [((s.t_orig[s.sfe>sfe_min]-pre_sim_time[i])/t_ff(s.initmass,s.R))[1:][np.diff(s.t_orig[s.sfe>sfe_min])>0] for i,s in enumerate(statlist)]
        eps_ff = [np.log10((np.diff(s.sfe[s.sfe>sfe_min])/np.diff(s.t_orig[s.sfe>sfe_min])*t_ff(s.initmass,s.R))[np.diff(s.t_orig[s.sfe>sfe_min])>0]) for s in statlist]     
        for i in range(len(eps_ff)):
            ind = np.isfinite(eps_ff[i])
            tnorm_alt[i] = tnorm_alt[i][ind]; eps_ff[i] = eps_ff[i][ind]
            #correct for rare cases where there is a big gap, so let's not allow more than 5 snaps
            endind = np.argmax(np.diff(np.arange(len(ind))[ind])>5)
            if endind:
                tnorm_alt[i] = tnorm_alt[i][:endind]; eps_ff[i] = eps_ff[i][:endind]
        evolplot(tnorm_alt, eps_ff,statlist,labels, r't/$\mathrm{t_{ff}}$', 'Log $\epsilon_\mathrm{ff,0}$', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize, xmax=maxtff,smooth_scale=0.01)
        # #SFE vs time       
        filename='sfe_t'+label
        sfe_min = np.max([1e-5, np.max([np.min(s.sfe[s.t>0]) for s in statlist])  ]); logsfe_min=np.log10(sfe_min);
        t = [s.t[s.sfe>sfe_min]*((pc/1)/Myr) for s in statlist]
        sfe = [np.log10(s.sfe[s.sfe>sfe_min]) for s in statlist]
        evolplot(t, sfe,statlist,labels, 't [Myr]', 'Log SFE', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, overtextfontsize=overtextfontsize, plot_fitlines=plot_fitlines, labelspacing=labelspacing,legendfontsize=legendfontsize)  
         #SFE vs time/t_ff
        filename='sfe_tnorm'+label
        t_norm_est = [np.log10(s.t[s.sfe>sfe_min]/t_ff(s.initmass,s.R)) for s in statlist]
        evolplot(t_norm_est, sfe,statlist,labels, r'Log $\tilde{t}$', 'Log SFE', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, overtextfontsize=overtextfontsize, plot_fitlines=False, labelspacing=labelspacing,legendfontsize=legendfontsize, noplot=True, xmax=maxtff) 
        leg=plt.legend(loc='best', fontsize=legendfontsize,labelspacing=labelspacing);
        #plot_fitlines=True
        #if plot_fitlines:
        if 1:
            fitx = 0.5*(plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            #find which to fit to
            fitind1 = -1; fitind2 = -1;
            if len(labels)>1:
                for i in range(len(labels)):
                    if 'box' in labels[i].lower() : fitind1 = i
                    if 'sphere' in labels[i].lower() : fitind2 = i
            fitpoint1 = [fitx,np.interp(fitx, t_norm_est[fitind1], sfe[fitind1])]
            fitpoint2 = [fitx,np.interp(fitx, t_norm_est[fitind2], sfe[fitind2])]
            fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle=':'),matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='--')]
            GDplot.add_line_to_plot(plt.gca(),2.0,point_to_include=fitpoint1,intercept=None,num=100,style=':',color='k')  
            GDplot.add_line_to_plot(plt.gca(),3.0,point_to_include=fitpoint2,num=100,style='--',color='k') 
            plt.legend(fitlines,[r'$\mathrm{SFE}\propto \tilde{t}^{2}$',r'$\mathrm{SFE}\propto \tilde{t}^{3}$'],loc=4, fontsize=legendfontsize,labelspacing=labelspacing)
        if not (extra_vlines is None):
            for i_val,val in enumerate(extra_vlines):
                plt.axvline(val,color=colors[i_val],linestyle='--')
        plt.gca().add_artist(leg)
        GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' );# plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
        plt.show()
        
        #Nsink vs time/t_ff      
        filename='Nsink_tnorm'+label       
        logNsink = [np.log10(s.Nsink[s.sfe>sfe_min]) for s in statlist]
        evolplot(t_norm_est, logNsink,statlist,labels, r'Log $\tilde{t}$', 'Log $N_\mathrm{sink}$', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, overtextfontsize=overtextfontsize, plot_fitlines=plot_fitlines, labelspacing=labelspacing,legendfontsize=legendfontsize, noplot=True, xmax=maxtff) 
        if plot_fitlines:
            leg=plt.legend(loc=2, fontsize=legendfontsize,labelspacing=labelspacing);
            fitx = 0.7*(plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            fitpoint1 = [fitx,np.interp(fitx, t_norm_est[-1], logNsink[-1])]
            fitpoint2 = [fitx,np.interp(fitx, t_norm_est[-2], logNsink[-2])]
            fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle=':'),matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='--')]
            GDplot.add_line_to_plot(plt.gca(),2.0,point_to_include=fitpoint1,intercept=None,num=100,style=':',color='k') 
            GDplot.add_line_to_plot(plt.gca(),3.0,point_to_include=fitpoint2,intercept=None,num=100,style='--',color='k')  
            plt.legend(fitlines,[r'$N_\mathrm{sink}\propto \tilde{t}^2$',r'$N_\mathrm{sink}\propto \tilde{t}^{3}$'],loc=4, fontsize=legendfontsize,labelspacing=labelspacing)
            plt.gca().add_artist(leg)
        if not (extra_vlines is None):
            for i_val,val in enumerate(extra_vlines):
                plt.axvline(val,color=colors[i_val],linestyle='--')
        GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' );#plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
        plt.show()
        
        #Mmax vs SF
        filename='Mmax_sfe'+label       
        logMmax = [np.log10(np.array([np.max(d.m,initial=0) for d in s.data])[s.sfe>sfe_min]) for s in statlist]
        evolplot(sfe, logMmax,statlist,labels, 'Log SFE', 'Log $M_\mathrm{max}$', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, overtextfontsize=overtextfontsize, plot_fitlines=plot_fitlines, labelspacing=labelspacing,legendfontsize=legendfontsize) 
        #Mmax vs time/t_ff  
        filename='Mmax_tnorm'+label 
        evolplot(t_norm_est, logMmax,statlist,labels, r'Log $\tilde{t}$', 'Log $M_\mathrm{max}$', filename,colors,styles,overtext,overtextcoord,plot_obs_IMF=plot_obs_IMF, overtextfontsize=overtextfontsize, plot_fitlines=plot_fitlines, labelspacing=labelspacing,legendfontsize=legendfontsize, noplot=True, xmax=maxtff) 
        if plot_fitlines:
            leg=plt.legend(loc=2, fontsize=legendfontsize,labelspacing=labelspacing);
            fitx1 = 0.63*(plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            fitx2 = 0.7*(plt.xlim()[1]-plt.xlim()[0])+plt.xlim()[0]
            fitpoint1 = [fitx,np.interp(fitx1, t_norm_est[-2], logMmax[-2])]
            fitpoint2 = [fitx,np.interp(fitx2, t_norm_est[-2], logMmax[-2])]
            fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle=':'),matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='--')]
            GDplot.add_line_to_plot(plt.gca(),0.5,point_to_include=fitpoint1,intercept=None,num=100,style=':',color='k') 
            GDplot.add_line_to_plot(plt.gca(),3.0,point_to_include=fitpoint2,intercept=None,num=100,style='--',color='k')  
            plt.legend(fitlines,[r'$M_\mathrm{max}\propto \tilde{t}^{1/2}$',r'$M_\mathrm{max}\propto \tilde{t}^{3}$'],loc=4, fontsize=legendfontsize,labelspacing=labelspacing)
            plt.gca().add_artist(leg)
        GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' );#plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
        plt.show()
        
        if not (overtext is None):
            if overtext!='':
                overtext+=', '
            if (comparison_SFE is None): 
                overtext += 'Final snapshot'
            else:
                if use_tff_instead_of_SFE:
                    overtext += r'At %3.2g $t_{ff}$'%(comparison_SFE)
                else:
                    overtext += 'SFE %3.2g'%(comparison_SFE)
    
    filename='imf'+label
    mass_logbins = [s.mass_logbins[s.IMF[ind]>0] for s,ind in zip(statlist,snap_ind)]
    if not imf_max_normalize:
        IMF = [np.log10((s.IMF[ind]/logmassbin_sizes)[s.IMF[ind]>0]) for s,ind in zip(statlist,snap_ind)]
        imf_label=imf_label_default
    else:
        IMF = [3+np.log10((s.IMF[ind]/logmassbin_sizes)[s.IMF[ind]>0]/np.max((s.IMF[ind]/logmassbin_sizes))) for s,ind in zip(statlist,snap_ind)]
        imf_label=imf_label_shifted
    plotfunc(mass_logbins,IMF,'',mass_label,imf_label,filename,labels=labels,\
                colors=colors,styles=styles,noplot=True,savefig=False,where_string='mid')
    if plot_fitlines and not plot_obs_IMF:
        fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='-.'),matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle=':')]
        GDplot.add_line_to_plot(plt.gca(),-1,offset=None,intercept=fitpos[0][0],num=100,style='-.',color='k')#,label='$dN/dlogM \propto M^{-1}$')
        GDplot.add_line_to_plot(plt.gca(),0,offset=None,intercept=fitpos[0][1],num=100,style=':',color='k')#,label='$dN/dlogM \propto M^0$')
    if plot_obs_IMF:
        fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='-.')]
        xvals=np.linspace(plt.xlim()[0],plt.xlim()[1],num=100)
        offset = fitpos[0][1]+plt.ylim()[1]
        plt.plot(xvals, np.log10(kroupa_IMF(xvals,offset=offset,log_units=True)),\
                 linestyle='-.',color='k')
        #original errors are 99% certainty, let's go for 95% instead
        err_mod = 2/3
        kroupa_err_plus = kroupa_IMF(xvals,offset=offset, log_units=True, hmass_slope=-2.3+kroupa_hmass_slope_err*err_mod,midmass_slope=-1.3+kroupa_midmass_slope_err*err_mod,lowmass_slope=-0.3+kroupa_lowmass_slope_err*err_mod)
        kroupa_err_plus /=  kroupa_IMF(kroupa_norm_point_logM,offset=offset, log_units=True, hmass_slope=-2.3+kroupa_hmass_slope_err*err_mod,midmass_slope=-1.3+kroupa_midmass_slope_err*err_mod,lowmass_slope=-0.3+kroupa_lowmass_slope_err*err_mod)/kroupa_IMF(kroupa_norm_point_logM,offset=offset,log_units=True)
        kroupa_err_min = kroupa_IMF(xvals,offset=offset, log_units=True, hmass_slope=-2.3-kroupa_hmass_slope_err*err_mod,midmass_slope=-1.3-kroupa_midmass_slope_err*err_mod,lowmass_slope=-0.3-kroupa_lowmass_slope_err*err_mod)
        kroupa_err_min /=  kroupa_IMF(kroupa_norm_point_logM,offset=offset, log_units=True, hmass_slope=-2.3-kroupa_hmass_slope_err*err_mod,midmass_slope=-1.3-kroupa_midmass_slope_err*err_mod,lowmass_slope=-0.3-kroupa_lowmass_slope_err*err_mod)/kroupa_IMF(kroupa_norm_point_logM,offset=offset,log_units=True)
        plt.fill_between(xvals,np.log10(kroupa_err_min),np.log10(kroupa_err_plus),alpha=0.3,color='k')
    if not (overtext is None):
        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left',\
                 verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
    dxlim=0; dylim=0.2*len(IMF); dylim0=0; 
    if plot_obs_IMF:
        dylim0-=0.2; dylim+=0.25;
    if not(xlim is None):
        plt.xlim(xlim)
        plt.xlim([plt.xlim()[0],plt.xlim()[1]+dxlim])
    plt.ylim([-0.2+dylim0-np.log10(np.mean(logmassbin_sizes)),plt.ylim()[1]+dylim])
    if normalize_masses:
        loc=3
        leg = plt.legend(loc=1, fontsize=legendfontsize,labelspacing=labelspacing);
    else:
        loc=3 #4
        leg = plt.legend(loc=1, fontsize=legendfontsize,labelspacing=labelspacing);
    if plot_fitlines and not plot_obs_IMF:
        plt.legend(fitlines,['$dN/dlogM \propto M^{-1}$','$dN/dlogM \propto M^0$'],loc=loc, fontsize=legendfontsize,labelspacing=labelspacing)
    if plot_obs_IMF:
        #plt.legend(fitlines,['Salpeter 1955','Kroupa 2002','Chabrier 2005'],loc=[0.55,0.51], fontsize=legendfontsize,labelspacing=labelspacing)
        plt.legend(fitlines,['Kroupa 2002'],loc=[0.55,0.9-0.1*len(IMF)], fontsize=legendfontsize,labelspacing=labelspacing)
    plt.gca().add_artist(leg)
    if addM50_to_plots:
        for i,s in enumerate(statlist):
            ind=snap_ind[i]
            plt.axvline(x=np.log10(s.mass_50[ind]),color=colors[i],linestyle=':')
            #Put symbol
            if put_symbol:
                m50_yval=np.interp(np.log10(s.mass_50[ind]),mass_logbins[i],IMF[i])
                plt.scatter(np.log10(s.mass_50[ind]),m50_yval,marker='o',c=[colors[i]],s=50)
    if (shade_low_masses and not(normalize_masses)):
        # top_pos_rel=0.8; top_pos=top_pos_rel*plt.ylim()[1]
        m_completeness_lim=np.max([s.minmass for s in statlist])
        if m_completeness_lim:
            # plt.axvline(x=np.log10(0.08),color='k',linestyle='--',alpha=0.5,ymax=top_pos_rel)
            # plt.arrow(np.log10(0.08), top_pos+0.05, -0.2, 0, head_width=0.05, head_length=0.03, linewidth=4, color='k', length_includes_head=True, alpha=0.5)
            # plt.text(np.log10(0.08),top_pos+0.2,"Brown dwarf limit", horizontalalignment='center',\
            #          verticalalignment='center', fontsize=overtextfontsize-3)
            xlim_temp=plt.xlim()
            plt.fill_betweenx(plt.ylim(),plt.xlim()[0],np.log10(m_completeness_lim),alpha=0.3)
            plt.text(np.log10(m_completeness_lim),plt.ylim()[0]+0.3*np.ptp(plt.ylim()),"Incompleteness", horizontalalignment='right', verticalalignment='center', fontsize=12,rotation='vertical',color='k')
            plt.xlim(xlim_temp)
    GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
    plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' );# plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
    plt.show()
    #Young star IMF
    if plot_young_IMF:
        young_agelimit = 5e5*yr/(pc/1); young_stage_limit=3; young_stage_agelimit = young_agelimit; 
        #Young star IMF
        filename='young_imf'+label
        mass_logbins = [s.mass_logbins for s in statlist]
        young_IMF_list =[]
        #Get IMF of young stars
        for s,ind in zip(statlist,snap_ind):
            d = s.data[ind]
            try:
                stage = d.val('ProtoStellarStage')
            except:
                stage = None
            if (stage is None):
                stage = np.zeros_like(d.m)
                young_timelimit = young_agelimit
            else:
                stage = d.val('ProtoStellarStage')
                young_timelimit = young_stage_agelimit
            stage_age = d.t - d.formation_time
            young_ind = (stage_age<young_timelimit) & (stage<=young_stage_limit)
            young_IMF, _ = np.histogram(np.log10(d.m[young_ind]), bins=s.mass_logbin_edges, density=False)
            if not imf_max_normalize:
                young_IMF = np.log10(young_IMF/logmassbin_sizes)
                imf_label='Log d$\mathrm{N_{young}}$/dlogM'
            else:
                young_IMF = 2+np.log10( (young_IMF/logmassbin_sizes)/np.max(young_IMF/logmassbin_sizes) )
                imf_label='Log d$\mathrm{N_{young}}$/dlogM [shifted]'
            
            young_IMF_list.append(young_IMF)    
        plotfunc(mass_logbins,young_IMF_list,'',mass_label,imf_label,filename,labels=labels,\
                    colors=colors,styles=styles,noplot=True,savefig=False,where_string='mid')
        if not(xlim is None):
            plt.xlim(xlim)
        if plot_fitlines and not plot_obs_IMF:
            fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='-.'),matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle=':')]
            GDplot.add_line_to_plot(plt.gca(),-1,offset=None,intercept=fitpos[0][0],num=100,style='-.',color='k')#,label='$dN/dlogM \propto M^{-1}$')
            GDplot.add_line_to_plot(plt.gca(),0,offset=None,intercept=fitpos[0][1],num=100,style=':',color='k')#,label='$dN/dlogM \propto M^0$')
        if plot_obs_IMF:
            fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='--'),\
                        matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='-.'),\
                        matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle=':')]
            xvals=np.linspace(plt.xlim()[0],plt.xlim()[1],num=100)
            GDplot.add_line_to_plot(plt.gca(),-1.35,offset=None,intercept=fitpos[0][0],num=100,style='--',color='k')
            plt.plot(xvals, np.log10(kroupa_IMF(xvals,offset=fitpos[0][1]+plt.ylim()[1],log_units=True)),\
                      linestyle='-.',color='k')
            plt.plot(xvals, np.log10(chabrier_IMF(xvals,offset=fitpos[0][1]+plt.ylim()[1]+0.85,log_units=True)),\
                      linestyle=':',color='k')
        if not (overtext is None):
            plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left',\
                      verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
        dxlim=0; dylim=0.75; dylim0=0; 
        if not (shade_low_masses):
            dylim=1.5; dylim0=-0.5
        if plot_obs_IMF:
            dylim0=-0.2; dylim=1.2;
        if not(xlim is None):
            plt.xlim(xlim)
            plt.xlim([plt.xlim()[0],plt.xlim()[1]+dxlim])
        plt.ylim([-0.2+dylim0-np.log10(np.mean(logmassbin_sizes)),plt.ylim()[1]+dylim])
        if normalize_masses:
            loc=3
            leg = plt.legend(loc=1, fontsize=legendfontsize,labelspacing=labelspacing);
        else:
            loc=3 #4
            leg = plt.legend(loc=1, fontsize=legendfontsize,labelspacing=labelspacing);
        if plot_fitlines and not plot_obs_IMF:
            plt.legend(fitlines,['$dN/dlogM \propto M^{-1}$','$dN/dlogM \propto M^0$'],loc=loc, fontsize=legendfontsize,labelspacing=labelspacing)
        if plot_obs_IMF:
            plt.legend(fitlines,['Salpeter 1955','Kroupa 2002','Chabrier 2005'],loc=[0.55,0.5], fontsize=legendfontsize,labelspacing=labelspacing)
        plt.gca().add_artist(leg)
        GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' );# plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
        plt.show()

    #Mass distribution
    if plot_mass_IMF:
        filename='mass_imf'+label
        mass_logbins_m = [s.mass_logbins[s.mass_PDF[ind]>0] for s,ind in zip(statlist,snap_ind)]
        #mass_PDF = [np.log10((s.mass_PDF[ind]/logmassbin_sizes/s.Msink[ind])[s.mass_PDF[ind]>0]) for s,ind in zip(statlist,snap_ind)]
        mass_PDF = [np.log10((s.mass_PDF[ind]/logmassbin_sizes)[s.mass_PDF[ind]>0]) for s,ind in zip(statlist,snap_ind)]
        plotfunc(mass_logbins_m,mass_PDF,'',mass_label,mass_imf_label,filename,labels=labels,\
                    colors=colors,styles=styles,noplot=True,where_string='mid')
        leg = plt.legend(loc=4, fontsize=legendfontsize,labelspacing=labelspacing);
        if not(xlim is None):
            plt.xlim(xlim)
        if plot_fitlines:
            fitlines = [matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle='-.'),matplotlib.lines.Line2D([0],[0],color='k',lw=2,linestyle=':')]
            GDplot.add_line_to_plot(plt.gca(),0,offset=None,intercept=fitpos[1][0],num=100,style='-.',color='k',label='$dM_{\mathrm{tot}}/dlogM \propto M^0$')
            GDplot.add_line_to_plot(plt.gca(),1.0,offset=None,intercept=fitpos[1][1],num=100,style=':',color='k',label='$dM_{\mathrm{tot}}/dlogM  \propto M$')
        if not (overtext is None):
            plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left',\
                     verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
        if addM50_to_plots:
            for i,s in enumerate(statlist):
                ind=snap_ind[i]
                plt.axvline(x=np.log10(s.mass_50[ind]),color=colors[i],linestyle=':')
                #Put symbol
                if put_symbol:
                    m50_yval=np.interp(np.log10(s.mass_50[ind]),mass_logbins_m[i],mass_PDF[i])
                    plt.scatter(np.log10(s.mass_50[ind]),m50_yval,marker='o',c=[colors[i]],s=50)
        if do_step_plots:
            plt.ylim([plt.ylim()[0]+5.0-0.2-np.log10(np.mean(logmassbin_sizes)),plt.ylim()[1]])
        GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        if plot_fitlines:
            plt.legend(fitlines,['$dM_{\mathrm{tot}}/dlogM \propto M^0$','$dN/dlogM \propto M^0$'],loc=[0.475,0.28], fontsize=legendfontsize,labelspacing=labelspacing)
        plt.gca().add_artist(leg)
        plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' );# plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
        plt.show()
    
        if do_CDFs:
            #CDF
            filename='cdf'+label
            CDF = [np.log10(s.CDF[ind]) for s,ind in zip(statlist,snap_ind)]
            CDF_mass = [s.CDF_mass[ind] for s,ind in zip(statlist,snap_ind)]
            plotfunc(CDF_mass,CDF,'',mass_label,'Log N($>M$)',filename,labels=labels,\
                        colors=colors,styles=styles,noplot=True,where_string='mid')
            if plot_fitlines:
                GDplot.add_line_to_plot(plt.gca(),-1,offset=None,intercept=fitpos[2][0],num=100,style='-.',color='k')
                params=[-1,np.max(np.concatenate(CDF_mass)),True]
                fitpos[2][1] += np.max(np.concatenate(CDF))-1
                GDplot.add_function_to_plot(plt.gca(),GDstat.logfunc,params,offset=fitpos[2][1],num=100,style=':',color='k')
            if not (overtext is None):
                plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left',\
                         verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
            if addM50_to_plots:
                for i,s in enumerate(statlist):
                    ind=snap_ind[i]
                    plt.axvline(x=np.log10(s.mass_50[ind]),color=colors[i],linestyle=':')
            if not(xlim is None):
                plt.xlim(xlim)
            if not(CDF_ylim is None):
                plt.ylim(CDF_ylim)
            if ((plt.xlim()[1]-plt.xlim()[0])>2.0):
                leg = plt.legend(loc=3, fontsize=legendfontsize,labelspacing=labelspacing);
            else:
                leg = plt.legend(loc=1, fontsize=legendfontsize,labelspacing=labelspacing);
            if plot_fitlines:
                plt.legend(fitlines,['$N(>M)\propto M^{-1}$','$N(>M)\propto \log(M_{\mathrm{max}}/M)$'],loc=1, fontsize=legendfontsize,labelspacing=labelspacing)
                plt.gca().add_artist(leg)
            GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' ); plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
            plt.show()
            #CMF
            filename='cmf'+label
            CMF = [np.log10(s.CMF[ind]) for s,ind in zip(statlist,snap_ind)]
            CMF_mass = [s.CMF_mass[ind] for s,ind in zip(statlist,snap_ind)]
            plotfunc(CMF_mass,CMF,'',mass_label,cum_mass_label,filename,labels=labels,\
                        colors=colors,styles=styles,noplot=True,where_string='mid')
            if plot_fitlines:
                if plot_fitlines is True:
                    params=[-1,np.max(np.concatenate(CMF_mass)),True]
                else:
                    params=[-1,np.max(CMF_mass[plot_fitlines-1]),True]
                fitpos[3][0] += np.max(np.concatenate(CMF))-1.0
                GDplot.add_function_to_plot(plt.gca(),GDstat.logfunc,params,offset=fitpos[3][0],num=100,style='-.',color='k',label='$M(>M)=\propto \log(M_{max}/M)$')
                if plot_fitlines is True:
                    params=[-1,np.max(np.concatenate(CMF_mass)),True]
                else:
                    params=[-1,np.max(CMF_mass[plot_fitlines-1]),True]
                fitpos[3][1] += np.max(np.concatenate(CMF))+1.5
                GDplot.add_function_to_plot(plt.gca(),GDstat.linfunc,params,offset=fitpos[3][1],num=100,style=':',color='k',label='$M(>M)=\propto (M_{max}-M)$')
            if not (overtext is None):
        #        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left',\
        #                 verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
                plt.text(overtextcoord[0], overtextcoord[1]-0.15, overtext, horizontalalignment='left',\
                         verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
            if addM50_to_plots:
                for i,s in enumerate(statlist):
                    ind=snap_ind[i]
                    plt.axvline(x=np.log10(s.mass_50[ind]),color=colors[i],linestyle=':')
            if not(xlim is None):
                plt.xlim(xlim)
            if not(CDF_ylim is None):
                plt.ylim(CDF_ylim)
            if ((plt.xlim()[1]-plt.xlim()[0])>2.0):
                plt.legend(loc=3, fontsize=legendfontsize,labelspacing=labelspacing);
            else:
                plt.legend(loc=1, fontsize=legendfontsize,labelspacing=labelspacing);
            GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' ); plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
            plt.show()

    if do_inv_CDFs:
        #Inverse CDF
        filename='inv_cdf'+label
        inv_CDF = [np.log10(s.inv_CDF[ind]) for s,ind in zip(statlist,snap_ind)]
        inv_CDF_mass = [s.inv_CDF_mass[ind] for s,ind in zip(statlist,snap_ind)]
        plotfunc(inv_CDF_mass,inv_CDF,'',mass_label,'Log N(<M)',filename,labels=labels,\
                    colors=colors,styles=styles,noplot=True,where_string='mid')
        if plot_fitlines:
            params=[-1,np.max([np.percentile(np.concatenate(inv_CDF),1),np.min(plt.xlim())]),True]
            GDplot.add_function_to_plot(plt.gca(),GDstat.invdifffunc,params,offset=fitpos[4][0],num=100,style='-.',color='k',label='$N(<M)\propto (M_{min}^{-1}-M^{-1})$')
            params=[1,np.max([np.percentile(np.concatenate(inv_CDF),1),np.min(plt.xlim())]),True]
            GDplot.add_function_to_plot(plt.gca(),GDstat.logfunc,params,offset=fitpos[4][1],num=100,style=':',color='k',label='$N(<M)\propto \log(M/M_{min})$')
        if not (overtext is None):
            plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left',\
                     verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
        if not(xlim is None):
            plt.xlim(xlim)
        if not(CDF_ylim is None):
            plt.ylim(CDF_ylim)
        if ((plt.xlim()[1]-plt.xlim()[0])>2.0):
            plt.legend(loc=3, fontsize=legendfontsize,labelspacing=labelspacing);
        else:
            plt.legend(loc=1, fontsize=legendfontsize,labelspacing=labelspacing);
        GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' ); plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
        plt.show()

        #Inverse CMF
        filename='inv_cmf'+label
        inv_CMF = [np.log10(s.inv_CMF[ind]) for s,ind in zip(statlist,snap_ind)]
        inv_CMF_mass = [s.inv_CMF_mass[ind] for s,ind in zip(statlist,snap_ind)]
        plotfunc(inv_CMF_mass,inv_CMF,'',mass_label,'Log $\mathrm{M_{tot}}(<M)$',filename,labels=labels,\
                    colors=colors,styles=styles,noplot=True,where_string='mid')
        if plot_fitlines:
            params=[1,np.max([np.percentile(np.concatenate(inv_CMF),2),np.min(plt.xlim())]),True]
            GDplot.add_function_to_plot(plt.gca(),GDstat.logfunc,params,offset=fitpos[5][0],num=100,style='-.',color='k',label='$M(<M)=\propto \log(M_{\mathrm{max}}/M)$')
            GDplot.add_line_to_plot(plt.gca(),1,offset=None,intercept=fitpos[5][1],num=100,style=':',color='k',label='$M(<M)\propto M^{1}$')
        if not (overtext is None):
            plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left',\
                     verticalalignment='center', transform=(plt.gca()).transAxes, fontsize=overtextfontsize)
        if not(xlim is None):
            plt.xlim(xlim)
        if not(CDF_ylim is None):
            plt.ylim(CDF_ylim)
        if ((plt.xlim()[1]-plt.xlim()[0])>2.0):
            plt.legend(loc=3, fontsize=legendfontsize,labelspacing=labelspacing);
        else:
            plt.legend(loc=1, fontsize=legendfontsize,labelspacing=labelspacing);
        GDplot.adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig(filename+'.png',dpi=150, bbox_inches='tight' ); plt.savefig(filename+'.pdf',dpi=150, bbox_inches='tight' )
        plt.show()
