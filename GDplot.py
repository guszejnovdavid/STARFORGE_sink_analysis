#Plotting routines
from __future__ import print_function
from matplotlib import pyplot as plt #for plotting
from matplotlib.ticker import NullFormatter
import matplotlib
import numpy as np

def CDF_at_value_from_PDF(bins,pdf,cdf_val):
    cdf=np.cumsum(pdf)/np.sum(pdf) #normalized CDF
    index=np.argmax(cdf>=cdf_val)
    return bins[index]

def adjust_font(lgnd=None, lgnd_handle_size=49, fig=None, ax_fontsize=14, labelfontsize=14,do_ticks=True ):
    if not (lgnd is None):
        for handle in lgnd.legendHandles:
            handle.set_sizes([lgnd_handle_size])
    if not (fig is None):
        ax_list = fig.axes
        for ax1 in ax_list:
            ax1.tick_params(axis='both', labelsize=ax_fontsize)
            ax1.set_xlabel(ax1.get_xlabel(),fontsize=labelfontsize)
            ax1.set_ylabel(ax1.get_ylabel(),fontsize=labelfontsize)
            if do_ticks:
                ax1.minorticks_on()
                ax1.tick_params(axis='both',which='both', direction='in',top=True,right=True)
            

def add_line_to_plot(axes,slope,offset=None,intercept=None,point_to_include=None,num=100,style='--',color='k',label=None):
    plt.sca(axes) #set current axes
    ylim=plt.ylim();xlim=plt.xlim(); xvals=np.linspace(xlim[0],xlim[1],num=num)
    if point_to_include is None:
        if offset is None:
            if intercept is None:
                offset=(np.mean(ylim)-(slope)*np.mean(xlim))
            else:
                if (slope==0):
                    offset=((np.max(ylim)-np.min(ylim))*intercept+np.min(ylim))
                else:
                    offset=np.mean(ylim)-(slope)*((np.max(xlim)-np.min(xlim))*intercept+np.min(xlim))
    else:
        offset = point_to_include[1]-point_to_include[0]*slope
    line=slope*xvals+offset
    plt.plot(xvals,line,style, c=color,label=label)
    plt.ylim(ylim)#reset y limits
    
def add_function_to_plot(axes,function,params,offset=0,num=100,style='--',color='k',label=None):
    plt.sca(axes) #set current axes
    ylim=plt.ylim();xlim=plt.xlim(); xvals=np.linspace(xlim[0],xlim[1],num=num)
    yvals=function(xvals,params)+offset
    plt.plot(xvals,yvals,style, c=color,label=label)
    plt.ylim(ylim)#reset y limits

def scatter_error_ind(Xmeans,Ymeans,Xvars,Yvars,Labels,Colors,title,xlabel,ylabel,filename, msize=200,\
                      fontsize=13, capsize=10,legendloc=4, noplot=False,markers='o',nosave=False):
    #Scatter plot with error bars, each point individually labeled and colored
    fig1, ax1 = plt.subplots(1,1)
    if (title != '' and title != None):
        plt.title(title)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    fig1.set_size_inches(6, 6)
    try:
        if (Xvars==None):
            Xvars=np.zeros(len(Xmeans))
        if (Yvars==None):
            Yvars=np.zeros(len(Xmeans))
    except:
        pass
    for i in range(len(Xmeans)):
        if (len(markers)==1):
            marker=markers
        else:
            marker=markers[i]
        plt.scatter(Xmeans[i],Ymeans[i],s=msize,c=Colors[i],marker=marker,label=Labels[i],zorder=10)
        if ( (Yvars[i]==0) and Xvars[i]>0):
            plt.errorbar(Xmeans[i],Ymeans[i],None,Xvars[i],ecolor=Colors[i],capsize=capsize,fmt='None',zorder=1)
        elif ( (Xvars[i]==0) and Yvars[i]>0):
            plt.errorbar(Xmeans[i],Ymeans[i],Yvars[i],None,ecolor=Colors[i],capsize=capsize,fmt='None',zorder=1)
        elif (Xvars[i]>0 and Yvars[i]>0):
            plt.errorbar(Xmeans[i],Ymeans[i],Yvars[i],Xvars[i],ecolor=Colors[i],capsize=capsize,fmt='None',zorder=1)
    plt.legend(loc=legendloc)
    if (noplot==False):
        plt.show()
    if (nosave==False):
        fig1.savefig(filename+'.png',dpi=150, bbox_inches='tight')
        #fig1.savefig(filename+'.pdf',dpi=150, bbox_inches='tight')

def scatter_error_bulk(X,Y,Xvars,Yvars,title,xlabel,ylabel,filename, msize=30, capsize=5,\
                       labels='', markers='o', colors='k',legendloc=4, noplot=False,\
                       elinewidth=None,graylines=False,overtext=None,overtextcoord=[0.5,0.5],\
                       overtextfontsize=30, fontsize=13,uplims=None):
    #Scatter plot with error bars, all the same color and no legend
    fig1, ax1 = plt.subplots(1,1)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    fig1.set_size_inches(6, 6)
    if (title != '' and title != None):
        plt.title(title)
    #Check is array or list of arrays
    if hasattr(X[0], "__iter__"):
        #get number of columns
        ncols=len(X)
        if (Xvars is None):
                Xvars=np.full(ncols,None)
        if (Yvars is None):
                Yvars=np.full(ncols,None)
        if (uplims is None):
                uplims=np.full(ncols,None)
        for i in range(ncols):
            if (labels==''):
                label=''
            else:
                label=labels[i]
            if (len(markers)==1):
                marker=markers
            else:
                marker=markers[i]
            if (len(colors)==1):
                color=colors
            else:
                color=colors[i]
            plt.scatter(X[i],Y[i],s=msize,c=color,marker=marker,label=label,zorder=10)
            if (graylines==True):
                color='0.6'
            plt.errorbar(X[i],Y[i],Yvars[i],Xvars[i],uplims=uplims[i],ecolor=color,capsize=capsize,fmt='None',elinewidth=elinewidth,zorder=1)
    else:
        plt.scatter(X,Y,s=msize,c=colors,marker=markers,label=labels,zorder=10)
        if (graylines==True):
            color='0.6'
        else:
            color=colors
        plt.errorbar(X,Y,Yvars,Xvars,ecolor=color,capsize=capsize,fmt='None',elinewidth=elinewidth,zorder=1,uplims=uplims)
    if (overtext!=None):
        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='center',\
                 verticalalignment='center', transform=ax1.transAxes, fontsize=overtextfontsize)
    if (labels!=''):
        plt.legend(loc=legendloc)
    if (noplot==False):
        plt.show()
    fig1.savefig(filename+'.png',dpi=150, bbox_inches='tight')
    #fig1.savefig(filename+'.pdf',dpi=150, bbox_inches='tight')


def scatter(X,Y,title,xlabel,ylabel,filename, msize=30, labels='', markers='o',colors='k',nolegend=False,\
            legendloc=4, noplot=False,savefig=True,overtext=None,overtextcoord=[0.5,0.5],overtextfontsize=30,\
                overtext_alignment='left', fontsize=13,alpha=1.0):
    #Scatter plot with error bars, all the same color and no legend
    fig1, ax1 = plt.subplots(1,1)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    fig1.set_size_inches(6, 6)
    if (title != '' and title != None):
        plt.title(title)
    #Check is array or list of arrays
    if hasattr(X[0], "__iter__"):
        #get number of columns
        ncols=len(X)
        for i in range(ncols):
            if (labels==''):
                label=''
            else:
                label=labels[i]
            if (len(markers)==1):
                marker=markers
            else:
                marker=markers[i]
            if (len(colors)==1):
                color=colors
            else:
                color=colors[i]
            plt.scatter(X[i],Y[i],s=msize,c=color,marker=marker,label=label,alpha=alpha)
    else:
        plt.scatter(X,Y,s=msize,c=colors,marker=markers,label=labels,alpha=alpha)
    if (labels!='' and nolegend==False):
        plt.legend(loc=legendloc)
    if (overtext!=None):
        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment=overtext_alignment,\
                 verticalalignment='center', transform=ax1.transAxes, fontsize=overtextfontsize)
    if (noplot==False):
        plt.show()
    if savefig:
        fig1.savefig(filename+'.png',dpi=150, bbox_inches='tight')
        #fig1.savefig(filename+'.pdf',dpi=150, bbox_inches='tight')
  
    
def step_plot(X,Y,title,xlabel,ylabel,filename, styles='-', labels='', colors='k',alpha=1.0,legendloc=4,xlim=[],ylim=[],\
         noplot=False, twiny=False, twiny_xlabel='',twiny_xticks=[],twiny_xticks_interp_func=[],where_string='pre',\
         twiny_xticks_0=None,overtext=None,overtextcoord=[0.5,0.5],overtextfontsize=30,overtext_alignment='left', nolegend=False,legendfontsize=12,\
         savefig=True):
    #Scatter plot with error bars, all the same color and no legend
    fig1, ax1 = plt.subplots(1,1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig1.set_size_inches(6, 6)
    if (title != '' and title != None):
        plt.title(title)
    #Check is array or list of arrays
    if hasattr(X[0], "__iter__"):
        #get number of columns
        ncols=len(X)
        for i in range(ncols):
            if (labels==''):
                label=''
            else:
                label=labels[i]
            if (len(styles)==1):
                style=styles
            else:
                style=styles[i]
            if (len(colors)==1):
                color=colors
            else:
                color=colors[i]
            plt.step(X[i],Y[i],linestyle=style,c=color,label=label,where=where_string)
    else:
        plt.step(X,Y,linestyle=styles,c=colors,label=labels,alpha=alpha,where=where_string)
    if len(xlim):
        plt.xlim(xlim)
    if len(ylim):
        plt.ylim(ylim)
    if (twiny):
        xlocs,_ = plt.xticks();xlim=plt.xlim();
        ax2=ax1.twiny() #create double axes
        plt.sca(ax2) #set current axes
        plt.xlabel(twiny_xlabel);plt.xlim(xlim);
        if (len(twiny_xticks_interp_func)>0):
            if hasattr(X[0], "__iter__"):
                twiny_xticks=np.round(np.interp(xlocs,X[0],twiny_xticks_interp_func),decimals=1)
            else:
                twiny_xticks=np.round(np.interp(xlocs,X,twiny_xticks_interp_func),decimals=1)
            if (twiny_xticks_0!=None):
                twiny_xticks[0]=twiny_xticks_0
        if (len(twiny_xticks)>0):
            plt.xticks(xlocs,twiny_xticks)
        plt.sca(ax1) #set current axes
        plt.tight_layout()
    if (labels!='' and not(nolegend)):
        plt.legend(loc=legendloc,fontsize=legendfontsize)
    if (overtext!=None):
        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment=overtext_alignment,\
                 verticalalignment='center', transform=ax1.transAxes, fontsize=overtextfontsize)
    if (noplot==False):
        plt.show()
    if savefig:
        fig1.savefig(filename+'.png',dpi=150, bbox_inches='tight')
        #fig1.savefig(filename+'.pdf',dpi=150, bbox_inches='tight')  

def plot(X,Y,title,xlabel,ylabel,filename, styles='-', labels='', colors='k',alpha=1.0,legendloc='best',xlim=[],ylim=[],\
         noplot=False, twiny=False, twiny_xlabel='',twiny_xticks=[],twiny_xticks_interp_func=[],\
         twiny_xticks_0=None,overtext=None,overtextcoord=[0.5,0.5],overtextfontsize=30,overtext_alignment='left', nolegend=False,legendfontsize=12,\
         savefig=True,where_string='pre',markers=None,markersize=0,linewidth=None,twiny_decimals=1):
    if linewidth is None:
        linewidth=matplotlib.rcParams['lines.linewidth']
    #Scatter plot with error bars, all the same color and no legend
    fig1, ax1 = plt.subplots(1,1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig1.set_size_inches(6, 6)
    if (title != '' and title != None):
        plt.title(title)
    #Check is array or list of arrays
    if hasattr(X[0], "__iter__"):
        #get number of columns
        ncols=len(X)
        for i in range(ncols):
            if (labels==''):
                label=''
            else:
                label=labels[i]
            if (len(styles)==1) and (not hasattr(styles[0], "__iter__")):
                style=styles
            else:
                style=styles[i]
            if (len(colors)==1) and (not hasattr(colors[0], "__iter__")):
                color=colors
            else:
                color=colors[i]
            if markers is None:
                marker=None
            else:
                if (len(markers)==1):
                    marker=markers
                else:
                    marker=markers[i]
            plt.plot(X[i],Y[i],style,c=color,label=label,alpha=alpha,marker=marker,markersize=markersize,linewidth=linewidth)
    else:
        plt.plot(X,Y,styles,c=colors,label=labels,alpha=alpha,marker=markers,markersize=markersize,linewidth=linewidth)
    if len(xlim):
        plt.xlim(xlim)
    if len(ylim):
        plt.ylim(ylim)
    if (twiny):
        xlocs,_ = plt.xticks();xlim=plt.xlim();
        ax2=ax1.twiny() #create double axes
        plt.sca(ax2) #set current axes
        plt.xlabel(twiny_xlabel);plt.xlim(xlim);
        if (len(twiny_xticks_interp_func)>0):
            if hasattr(X[0], "__iter__"):
                twiny_xticks=np.round(np.interp(xlocs,X[0],twiny_xticks_interp_func),decimals=twiny_decimals)
            else:
                twiny_xticks=np.round(np.interp(xlocs,X,twiny_xticks_interp_func),decimals=twiny_decimals)
            if (twiny_decimals==0):
                twiny_xticks=np.int32(twiny_xticks)
            if (twiny_xticks_0!=None):
                twiny_xticks[0]=twiny_xticks_0
        if (len(twiny_xticks)>0):
            plt.xticks(xlocs,twiny_xticks)
        plt.sca(ax1) #set current axes
        plt.tight_layout()
    if (labels!='' and not(nolegend)):
        plt.legend(loc=legendloc,fontsize=legendfontsize)
    if (overtext!=None):
        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment=overtext_alignment,\
                 verticalalignment='center', transform=ax1.transAxes, fontsize=overtextfontsize)
    if (noplot==False):
        plt.show()
    if savefig:
        fig1.savefig(filename+'.png',dpi=150, bbox_inches='tight')
        #fig1.savefig(filename+'.pdf',dpi=150, bbox_inches='tight')  
        
def color_under_plot(x,y,z,cmapname='afmhot', normalize_z=True, noplot=False,fontsize=13):
    # Select a color map
    cmap = plt.get_cmap(cmapname)
    normalize = matplotlib.colors.Normalize(vmin=z.min(), vmax=z.max())
    # The plot
    #plt.plot(x, y)
    for i in range(len(x) - 1):
        if normalize_z:
            znorm = normalize(z[i])
        else:
            znorm=z[i]
        plt.fill_between([x[i], x[i+1]], [y[i], y[i+1]], color=cmap(znorm))  
    if (noplot==False):
        plt.show()
          
   
def histogram(data,weights,nbin,filename,xlabel='',ylabel='',loglogplot=False,logplot=False,noplot=False,\
              fontsize=13,normalized=True):
    #Scatter plot with error bars, all the same color and no legend
    fig1, ax1 = plt.subplots(1,1)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    fig1.set_size_inches(6, 6)
    if logplot or loglogplot:
        if not(weights is None):
            weights=weights[data>0]
        data=np.log10(data[data>0])
    hist, bin_edges = np.histogram(data,nbin,weights=weights,density=normalized)
    bins=(bin_edges[:-1]+bin_edges[1:])/2.0
    if loglogplot:
        ind=hist>0
        hist=np.log10(hist[ind])
        hist=hist-np.min(hist)
        bins=bins[ind] 
    plt.bar(bins,hist,width=np.mean(bin_edges[:-1]-bin_edges[1:]))
    plt.ylim([0,np.max(hist)])
    if (noplot==False):
        plt.show()
    fig1.savefig(filename+'.png',dpi=150, bbox_inches='tight')
    #fig1.savefig(filename+'.pdf',dpi=150, bbox_inches='tight')   
    
def scatter_with_histograms(X,Y,xlabel,ylabel,filename,nbins=20,xlim=None,ylim=None,\
                            msize=30, capsize=5, markers='o', colors='k',cmap='viridis',\
                            overtext=None,overtextcoord=[0.5,0.5], loghist=False,\
                            histx_color='b',histy_color='b',\
                            colorbar=False, cbar_limits=None,cbar_tick_labels=None,cbar_label='', fontsize=15 ): 
    # Start making the plot
    # definitions for the axes, sets size of subfigures
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    #Limits
    if not hasattr(xlim, "__iter__"):
        xlim=np.array([np.min(X),np.max(X)])
        ylim=np.array([np.min(Y),np.max(Y)])
    # creates the axes
    fig = plt.figure(1, figsize=(8, 8))
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)# We want plots and no labels on the overlapping part of the histograms
    nullfmt = NullFormatter()
    #Set log scale for histogram if needed
    if loghist:
        axHistx.set_yscale("log")
        axHisty.set_xscale("log")
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
#    # diagonal line
#    axScatter.plot([xlim[0],xlim[1]],[ylim[0],ylim[1]],'--',c='black',lw=1.5,alpha=0.25)
    #Scatter plot
    im = axScatter.scatter(X , Y ,marker=markers, c=colors,edgecolors='black',cmap=cmap);
    #Set limits
    axScatter.set_xlim((xlim[0], xlim[1])); axScatter.set_ylim((ylim[0], ylim[1]))
    #Set label
    axScatter.set_xlabel(xlabel, fontsize=fontsize); axScatter.set_ylabel(ylabel, fontsize=fontsize);
#    #Add colorbar
    if colorbar:
        if not hasattr(cbar_limits, "__iter__"):
            cbar_limits=np.array([np.min(colors),np.max(colors)])
        cb = plt.colorbar(im,boundaries=np.linspace(cbar_limits[0], cbar_limits[1], 100, endpoint=True)) # 'boundaries' is so the drawn colorbar ranges between the limits you want (rather than the limits of the data)
        cb.set_clim(cbar_limits[0],cbar_limits[1]) # this only redraws the colors of the points, the range shown on the colorbar itself will not change with just this line alone (see above line)
        if hasattr(cbar_tick_labels, "__iter__"):
            nticks=len(cbar_tick_labels)
            cb.set_ticks(np.linspace(cbar_limits[0], cbar_limits[1], nticks, endpoint=True))
            cb.set_ticklabels(cbar_tick_labels)
        else:
            cb.set_ticks(np.linspace(cbar_limits[0], cbar_limits[1], 5, endpoint=True))
        if (cbar_label != '' and cbar_label != None):
            cb.set_label(cbar_label, fontsize=fontsize)
    axHistx.hist(X, bins=np.linspace(xlim[0],xlim[1],nbins),edgecolor='white',color=histx_color);
    axHisty.hist(Y, bins=np.linspace(ylim[0],ylim[1],nbins),edgecolor='white',color=histy_color, orientation='horizontal');
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    #Plot overtext
    if (overtext!=None):
        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='center', verticalalignment='center', transform=axScatter.transAxes, fontsize=30)
    #Save figure
    fig.savefig(filename+'.png',dpi=150, bbox_inches='tight')
    fig.savefig(filename+'.pdf',dpi=150, bbox_inches='tight')
    


def hist_comb_plot(xvals,yvals,x_edges,ybins,xlabel,ylabel,filename,weights=0,plot_image=True,altxvals=[],altxlabel='',\
                   overtext='',overtextcoord=[0.5,0.5],ypercentile_limits=[0,99.5],maxdexstretch=2,\
                    cmap='viridis', linecolor='white'):
    #Plots a 2D histogram with median an quartiles overplottd on it. Note if xedges or ybin is a number then it setsthe number of bins
    import GDstat #needed to actually calculate the statistics
    sortind=xvals.argsort()
    xvals=xvals[sortind]
    yvals=yvals[sortind]
    if hasattr(weights, "__iter__"):
        weights=weights[sortind]
    #Create histogram and calculate averages
    hist,y_stat_val,y_val_plus,y_val_minus,xbins,ybins = GDstat.hist2D_with_stat(xvals,yvals,x_edges=x_edges,\
                                                                                 ybins=ybins,weights=weights,\
                                                                                 ypercentile_limits=ypercentile_limits)
    fig1, ax1 = plt.subplots(1,1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig1.set_size_inches(6, 6)
    histminval=(10.0**(-maxdexstretch))*np.max(hist[np.nonzero(hist)])
    norm_hist=GDstat.normalize_array(np.log10(hist+histminval))
    x_edges=GDstat.bin_to_edge(xbins);y_edges=GDstat.bin_to_edge(ybins);
    ax1.imshow(np.flipud(np.transpose(norm_hist)), cmap=cmap,vmin=0, vmax=1.1,\
                extent=[x_edges[0],x_edges[-1],y_edges[0],y_edges[-1]],\
               aspect=(x_edges[-1]-x_edges[0])/(y_edges[-1]-y_edges[0]))
    #Overplot median and quartiles at every bin
    plt.plot(xbins,y_stat_val,'-',color=linecolor)
    plt.plot(xbins,y_val_plus,'--',color=linecolor)
    plt.plot(xbins,y_val_minus,'--',color=linecolor)
    #Text
    plt.text(overtextcoord[0], overtextcoord[1], overtext,color=linecolor, \
             horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=25)
    xlocs,_ = plt.xticks();
    xlocs=xlocs[(xlocs>=x_edges[0]) & (xlocs<=x_edges[-1])]
    #Create top axis
    if len(altxvals):
        altxticks=np.interp(xlocs,xvals,altxvals[sortind])
        ax2=ax1.twiny() #create double
        plt.sca(ax2) #set current axes
        plt.xlim([x_edges[0],x_edges[-1]])
        plt.xlabel(altxlabel)
        plt.xticks(xlocs,np.round(altxticks,decimals=1) )
        plt.tight_layout()
    #impose  y limits
    plt.ylim([y_edges[0],y_edges[-1]])
    if plot_image:
        plt.show()
    plt.tight_layout()
    fig1.savefig(filename+'.png',dpi=150, bbox_inches='tight')
    #fig1.savefig(filename+'.pdf',dpi=150, bbox_inches='tight')