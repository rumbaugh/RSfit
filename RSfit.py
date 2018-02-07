import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_rs(ax, y0, m, width, xspace=None, plotcenter=False):
    # Plot red sequence
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if np.shape(xspace) == ():
        xspace = np.linspace(xlim[0], xlim[1])
    ydummy = rs_model(xspace, y0, m)
    ax.plot(xspace, ydummy + 0.5*width, ls='--', lw=2, color='r')
    ax.plot(xspace, ydummy - 0.5*width, ls='--', lw=2, color='r')
    if plotcenter: ax.plot(xspace, ydummy, ls='--', lw=2, color='r')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    plt.show(block=False)

def rs_model(x, y0, m):
    return m*x + y0

def rsinit(ax):
    initx, inity = get_inp_pnt()
    ax.scatter([initx], [inity], marker='x', color='r', s=32)
    plt.show(block=False)
    initwid = get_inp_wid()
    ax.axhline(inity + 0.5*initwid, ls='--', lw=2, color='r')
    ax.axhline(inity - 0.5*initwid, ls='--', lw=2, color='r')
    plt.show(block=False)
    initslope = get_inp_slope()
    return initx, inity, initslope, initwid

def get_inp_wid():
    doneflag = False
    while not(doneflag):
        inp_wid = raw_input("\nSet the width for the initial red sequence fit (should be the full width):\n")
        try:
            wid = float(inp_wid)
            doneflag = True
        except:
            print "Input must be a float: %s"%inp_wid
    return wid

def get_inp_slope(set_negative=True):
    doneflag = False
    while not(doneflag):
        inp_m = raw_input("\nSet the slope for the initial red sequence fit (should be negative):\n")
        try:
            m = float(inp_m)
            if ((set_negative) & (m>0)): m *= -1
            doneflag = True
        except:
            print "Input must be a float: %s"%inp_m
    return m

def get_inp_pnt():
    doneflag = False
    while not(doneflag):
        inp_pnt = raw_input("\nPick a point for the initial red sequence line to go through:\n")
        tmpstr = inp_pnt.split(', ')
        if len(tmpstr)==1: 
            tmpstr = inp_pnt.split(' ')
        if len(tmpstr)==1: 
            tmpstr = inp_pnt.split(',')
        if (len(tmpstr)!= 2): 
            print "Invalid input: %s"%inp_pnt
        else:
            try:
                x = float(tmpstr[0])
                y = float(tmpstr[1])
                doneflag = True
            except:
                print "Inputs must be floats: %s"%inp_pnt
    return x, y

def one_color_cmap(cmap=pl.cm.gray):
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:,-1] = np.append(np.ones(1), np.zeros(cmap.N-1))
    my_cmap = ListedColormap(my_cmap)
    return my_cmap
