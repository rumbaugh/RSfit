import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import linregress
from scipy.interpolate import interp1d
from sklearn import mixture
import pydl.pydlutils.spheregroup
from set_spec_dict import set_spec_dict
from supercolors import *


class CMD:
    def __init__(self, specpath = None, field = None, show = False):
        self.spec_dict = set_spec_dict() #Initialize dictionary with spectroscopy info
        if specpath == None: specpath = self.spec_dict['basepath']
        self.specpath = specpath
        self.show = show

    def set_field(self, field, checkfield = True):
        if checkfield:
            # Checks if field is a valid option
            if not(field in self.spec_dict.keys()):
                print "{} is not a valid field".format(field)
                return
        self.field = field
        if checkfield: return True

    def analyze_field(self, field, checkfield = True, RSmodel_stepwait = 0):
        checkedfield = self.set_field(field, checkfield)
        if checkfield:
            if not(checkedfield): return
        self.load_spec()
        self.load_phot()
        self.match_spec2phot()
        self.set_CMD_mags()
        self.plot_CMD()
        self.setRSinit(numsig = 2)
        self.RSsigclip(RSmodel_stepwait)

    def load_spec(self, prioritizeACS = True, cut_by_q = True, cut_by_z = True, maxmag = 40, minmag = None):
        # Loads specotrscopic file
        # Information paths is contained in spec_dict
        self.speccat = '{}{}'.format(self.specpath,self.spec_dict[self.field]['file'])
        self.specdf = pd.read_csv(self.speccat, usecols = self.spec_dict[self.field]['coltup'], delim_whitespace = True)
        try:
            self.specdf['ID']
        except KeyError:
            if ((self.field == 'cl1604') & prioritizeACS):
                self.specdf['ID'] = self.specdf['ACS_ID']
            elif self.field == 'cl1604':
                self.specdf['ID'] = self.specdf['#LFC_ID']
            else:
                self.specdf['ID'] = self.specdf['#ID']
        # RA/Dec columns are indexed by LFC/ACS. Need to pick one (in the
        # case of cl1604), or just use LFC.
        if prioritizeACS:
            try:
                self.set_RADEC_col('ACS')
            except KeyError:
                self.set_RADEC_col('LFC')
        else:
            self.set_RADEC_col('LFC')
        # Make cuts to spectroscoptic data frame
        if cut_by_q: self.cut_spec_by_q()
        if cut_by_z: self.cut_spec_by_z()
        if ((maxmag != None) & (minmag != None)): self.cut_spec_by_mag(maxmag, minmag)

    def cut_spec_by_mag(self, maxmag = 40, minmag = None):
        # Cuts spectroscopic catalog by outlier magnitudes
        self.specdf = self.specdf[(self.specdf.r < maxmag) & (self.specdf.i < maxmag) & (self.specdf.z < maxmag)]
        if minmag != None: self.specdf = self.specdf[(self.specdf.r > minmag) & (self.specdf.i > minmag) & (self.specdf.z > minmag)]

    def cut_spec_by_q(self, goodQ = [3, 4]):
        # Cuts spectroscopic catalog by Q (the quality flag)
        # Good quality flags are 3 and 4. -1 is technically
        # good, too, but it designates stars, which are not
        # applicable to a CMD
        self.specdf = self.specdf[np.in1d(self.specdf.Q, goodQ)]

    def cut_spec_by_z(self, zLB = None, zUB = None):
        # Cuts spectroscopic catalog by redshift, between zLB
        # and zUB. Uses bounds from spec_dict if they aren't provided
        if zLB == None: zLB = self.spec_dict[self.field]['z'][1]
        if zUB == None: zUB = self.spec_dict[self.field]['z'][2]
        self.specdf = self.specdf[(self.specdf.redshift >= zLB) & (self.specdf.redshift <= zUB)]
    
    def set_RADEC_col(self, source): #called by load_spec
        try:
            self.specdf['RA'], self.specdf['Dec'] = self.specdf['{}_RA'.format(source)], self.specdf['{}_Dec'.format(source)]
        except KeyError:
            self.specdf['RA'], self.specdf['Dec'] = self.specdf['{}_RA'.format(source)], self.specdf['{}_DEC'.format(source)]

    def load_phot(self, photpath = '/home/rumbaugh/Chandra/photcats/', nameoverride = None, photdict = {'names': ('ID', 'dum1', 'dum2', 'RA', 'Dec', 'r', 'r_err', 'i', 'i_err', 'z', 'z_err'), 'formats': ('|S64', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')}):
        # Loads photometric data file
        if nameoverride != None:
            self.photcat = '{}{}'.format(photpath, nameoverride)
        else:
            self.photcat = '{}{}_rizdata.corr.gz'.format(photpath, self.field)
        photnames = np.array(photdict['names'])
        cols2use = self.set_cols2use(len(photnames))
        self.photdf = pd.read_csv(self.photcat, names = photnames, delim_whitespace = True, usecols = cols2use)        

    def set_cols2use(self, n):
        #Creates tuple with ascending integers for use in loading files
        cols2use = (0,)
        for i in range(1,n): cols2use = cols2use + (i,)
        return cols2use

    def match_spec2phot(self, match_rad = 0.5/3600, merge = True):
        # Matches spectroscopic and photometric catalogs
        # using pydl.pydlutils.spheregroup.spherematch
        try:
            self.specdf, self.photdf
        except AttributeError:
            print "Must set specdf and photdf"
            return
        self.match_inds = pydl.pydlutils.spheregroup.spherematch(self.photdf.RA.values, self.photdf.Dec.values, self.specdf.RA.values, self.specdf.Dec.values, match_rad)
        matched_phot = self.photdf.iloc[self.match_inds[0]]
        matched_phot.reset_index(inplace = True)
        self.orig_specdf, self.specdf = self.specdf, self.specdf.iloc[self.match_inds[1]]
        self.specdf.reset_index(inplace = True)
        if merge:
            self.specdf = pd.merge(self.specdf, matched_phot, left_index = True, right_index = True, suffixes = ('', '_phot'))
        else:
            return matched_phot
        
    def set_CMD_mags(self, use_supercolors = True, z_threshold = 1.0, ACS_override = True):
        # Sets the magnitudes to be used for making CMDs
        # Options are either supercolors, or r/i/z, depending
        # on the redshift threshold set by z_threshold
        try:
            self.specdf.r_err, self.specdf.i_err, self.specdf.z_err
        except AttributeError:
            if (not((self.field == 'cl1604') & ACS_override)):
                print "Must set photometric errors"
                return
        if ((self.field == 'cl1604') & ACS_override):
            self.m_b, self.m_r = self.specdf.F606W, self.specdf.F814W
            self.m_b_name, self.m_r_name = 'F606W', 'F814W'
        elif use_supercolors:
            self.m_b, self.m_r, self.m_b_err, self.m_r_err = calc_supercolor_mags(self.specdf.r, self.specdf.i, self.specdf.z, self.specdf.r_err, self.specdf.i_err, self.specdf.z_err, self.specdf.redshift)
            self.m_b_name, self.m_r_name = r'$M_{blue}$', r'$M_{red}$'
        else:
            if self.spec_dict[field]['z'][0] > z_threshold:
                self.m_b, self.m_r, self.m_b_err, self.m_r_err = self.specdf.i, self.specdf.z, self.specdf.i_err, self.specdf.z_err 
                self.m_b_name, self.m_r_name = 'i', 'z'
            else:
                self.m_b, self.m_r, self.m_b_err, self.m_r_err = self.specdf.r, self.specdf.i, self.specdf.r_err, self.specdf.i_err 
                self.m_b_name, self.m_r_name = 'r', 'i'


    def plot_CMD(self, show = True, figure = 1, clear = True):
        # Plots a color-magnitude diagram
        try:
            self.m_b, self.m_r
        except AttributeError:
            print "Must set m_b and m_r"
            return
        try:
            self.ax
        except AttributeError:
            self.fig = plt.figure(figure)
            self.ax = self.fig.add_subplot(111)
        if clear: self.ax.cla()
        set_plot_params()
        self.ax.scatter(self.m_r, self.m_b - self.m_r)
        self.ax.set_xlabel(self.m_r_name)
        self.ax.set_ylabel('{} - {}'.format(self.m_r_name, self.m_b_name))
        self.ax.set_title(self.field.upper())
        if show: plt.show(block = False)

    def plotRSO(self, plotRSbounds = True, RSOstep = 0.25, show = True, figure = 2, clear = True):
        # Plots histogram of red sequence offsets
        try:
            self.m_b, self.m_r, self.y0, self.m, self.wid
        except AttributeError:
            print "Must set magnitudes and RS parameters"
            return
        try:
            self.RSOax
        except AttributeError:
            self.RSOfig = plt.figure(figure)
            self.RSOax = self.RSOfig.add_subplot(111)
        if clear: self.RSOax.cla()
        set_plot_params()
        RSOs = (RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)) / (self.wid * 0.5)
        # Creates a nice range for plotting, using bin widths 
        # equal to RSOstep
        rangeLB, rangeUB = np.floor(np.min(RSOs)/RSOstep)*RSOstep, np.ceil(np.max(RSOs)/RSOstep)*RSOstep
        nbins = int((rangeUB - rangeLB)/RSOstep)
        # Plot histogram of RSOs
        self.RSOax.hist(RSOs, bins = nbins, range = (rangeLB, rangeUB))
        # Plot bounds of red sequence
        if plotRSbounds:
            self.RSOax.axvline(-1, ls = 'dashed', lw = 2, color = 'k')
            self.RSOax.axvline(+1, ls = 'dashed', lw = 2, color = 'k')
        self.ax.set_xlabel('Red Sequence Offset')
        self.ax.set_ylabel('Number of galaxies')
        self.ax.set_title(self.field.upper())
        
    def setRSinit(self, figure = 1, numsig = None):
        # Uses user input to initialize RS parameters/fitting region
        try:
            self.ax
        except AttributeError:
            print "Plot CMD first"
            return
        if numsig == None:
            self.numsig = 3
            if self.field in ['cl1324', 'cl1604']: self.numsig = 2
        else:
            self.numsig = numsig
        x, y, m, wid = RSinit(self.ax) #Initial guess
        changeflag = False
        while not(changeflag):
            #Tweak parameters until you're satisfied
            self.plot_CMD()
            self.y0, self.m, self.wid = m*(-x) + y, m, wid
            plot_RS(self.ax, self.y0, self.m, self.wid, plotcenter = True)
            inp_c = raw_input("\nWhat to change next? (N)ew point, (S)lope, (W)idth, (D)one\n")
            tmpL = inp_c[0].lower()
            if tmpL=='n':
                x, y = get_inp_pnt()
            elif tmpL=='s':
                m = get_inp_slope()
            elif tmpL=='w':
                wid = get_inp_wid()
            elif tmpL=='d':
                inp_chk = raw_input("\nReally done?\n")
                if inp_chk[0].lower()=='y':
                    changeflag = True
                    self.y0 = m*(-x) + y
                    self.m, self.wid = m, wid
                    self.sig = wid/(2.*self.numsig)
            else:
                print "Invalid input: %s"%inp_c
            plt.show(block=False)

    def RSsigclip(self, stepwait = 0):
        #Does sigma-clipping/linear fit algorithm to fit RS
        try:
            self.y0, self.m, self.wid, self.sig
        except AttributeError:
            print "Run setRSinit first"
            return
        # Do initial linear fit to RS model
        onRS = np.abs(RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)) <= 0.5*self.wid
        RSOs = RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)
        self.m, self.y0, rvalue, pvalue, stderr = linregress(self.m_r[onRS], self.m_b[onRS] - self.m_r[onRS])
        # Do 1-D sigma clip in color space
        RSOs = RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)
        onRS = np.abs(RSOs) <= 0.5*self.wid
        prevonRS = np.copy(onRS)
        doclip = True
        while doclip:
            onRS = np.abs(RSOs)  <= np.std(RSOs[onRS]) * self.numsig
            # End when there is no change in RS membership
            if np.count_nonzero(onRS != prevonRS) > 0: doclip = False
            if stepwait > 0:
                # Optional step-by-step plotting
                self.plot_CMD()
                plot_RS(self.ax, self.y0, self.m, self.numsig*2*np.std(RSOs[onRS]), plotcenter = False)
                time.sleep(stepwait)
            prevonRS = np.copy(onRS)
        self.sig = np.std(RSOs[onRS])
        self.wid = 2*self.numsig*self.sig
        self.plot_CMD()
        plot_RS(self.ax, self.y0, self.m, self.wid, plotcenter = False)

    def RSmodel(self, stepwait = 0):
        #Does sigma-clipping/linear fit algorithm to fit RS
        try:
            self.y0, self.m, self.wid, self.sig
        except AttributeError:
            print "Run setRSinit first"
            return
        dofit = True
        onRS = np.abs(RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)) <= self.wid
        prevonRS = np.copy(onRS)
        while dofit:
            self.m, self.y0, rvalue, pvalue, stderr = linregress(self.m_r[onRS], self.m_b[onRS] - self.m_r[onRS])
            onRS = np.abs(RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)) <= self.wid
            self.sig = np.std(RS_model(self.m_r[onRS], self.y0, self.m) - (self.m_b[onRS] - self.m_r[onRS]))
            self.wid = (2.*self.numsig)*self.sig
            onRS = np.abs(RS_model(self.m_r, self.y0, self.m) - (self.m_b -self.m_r)) <= self.wid
            # End when there is no change in RS membership
            if np.count_nonzero(onRS != prevonRS) > 0: dofit = False 
            if stepwait > 0:
                self.plot_CMD()
                plot_RS(self.ax, self.y0, self.m, self.wid, plotcenter = False)
                time.sleep(stepwait)
            prevonRS = np.copy(onRS)

    def RSmixture(self, covtype = 'full'):
        try:
            self.m_b, self.m_r
        except AttributeError:
            print "Set magnitudes first"
            return
        X = pd.DataFrame({'m_r': self.m_r, 'color': self.m_b - self.m_r})[['m_r','color']]
        self.GMM = mixture.GaussianMixture(2, covariance_type = covtype)
        self.GMM.fit(X)

    def plot_RSmixture(self):
        # Plot Gaussians
        try:
            self.GMM, self.ax
        except AttributeError:
            print "run RSmixture and created CMD plot first"
        # Predict which Gaussian correspons to RS
        if self.GMM.means_[:,1][1] == np.min(self.GMM.means_[:,1]):
            cov_index = [0,1]
        else:
            cov_index = [1,0]
        for cov, Gmean, color in zip(self.GMM.covariances_[cov_index], self.GMM.means_[cov_index], ['red','blue']):
            v, w = linalg.eigh(cov)
            angle = 180. * np.arctan2(w[0][1], w[0][0])/ np.pi
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            for v in [v, 2*v, 3*v]:
                ell = mpl.patches.Ellipse(Gmean, v[0], v[1], 180. + angle, color=color, fill = False, lw = 2)
                ell.set_clip_box(self.ax.bbox)
                self.ax.add_artist(ell)
                
                

def set_plot_params():
    # Standard plotting parameters
    plt.rc('axes',linewidth=2)
    plt.fontsize = 14
    plt.tick_params(which='major',length=8,width=2,labelsize=14)
    plt.tick_params(which='minor',length=4,width=1.5,labelsize=14)
