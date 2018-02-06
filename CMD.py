import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")#, category=DeprecationWarning)
    import numpy as np
    import pandas as pd
    import time
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy import linalg
    from scipy.stats import linregress
    from scipy.interpolate import interp1d
    from sklearn import mixture, cluster
    import pydl.pydlutils.spheregroup
    from set_spec_dict import set_spec_dict
    from supercolors import *
    from RSfit import *


class CMD:
    def __init__(self, specpath=None, field=None, show=False):
        self.spec_dict = set_spec_dict() #Initialize dictionary with spectroscopy info
        if specpath == None: 
            specpath = self.spec_dict['basepath']
        self.specpath = specpath
        self.show = show

    def set_field(self, field, checkfield=True):
        if checkfield:
            # Checks if field is a valid option
            if not(field in self.spec_dict.keys()):
                print "{} is not a valid field".format(field)
                return
        self.field = field
        if checkfield: return True

    def analyze_field(self, field, checkfield=True, rs_model_stepwait=0, loadphot=False):
        checkedfield = self.set_field(field, checkfield)
        if checkfield:
            if not(checkedfield): return
        self.load_spec()
        if loadphot:
            self.load_phot()
            self.match_spec2phot()
        self.set_cmd_mags()
        self.plot_cmd()
        self.set_rs_init(num_sigma=2)
        self.rs_sigclip(rs_model_stepwait)

    def load_spec(self, prioritize_acs=True, cut_by_q=True, cut_by_z=True, maxmag=40, minmag=None):
        # Loads specotrscopic file
        # Information on paths is contained in spec_dict
        self.speccat = '{}{}'.format(self.specpath,self.spec_dict[self.field]['file'])
        self.specdf = pd.read_csv(self.speccat, usecols=self.spec_dict[self.field]['coltup'], delim_whitespace=True)
        try:
            self.specdf['ID']
        except KeyError:
            if ((self.field == 'cl1604') & prioritize_acs):
                self.specdf['ID'] = self.specdf['ACS_ID']
            elif self.field == 'cl1604':
                self.specdf['ID'] = self.specdf['#LFC_ID']
            else:
                self.specdf['ID'] = self.specdf['#ID']
        # RA/Dec columns are indexed by LFC/ACS. Need to pick one (in the
        # case of cl1604), or just use LFC.
        if prioritize_acs:
            try:
                self.set_radec_col('ACS')
            except KeyError:
                self.set_radec_col('LFC')
        else:
            self.set_radec_col('LFC')
        # Make cuts to spectroscoptic data frame
        if cut_by_q: self.cut_spec_by_q()
        if cut_by_z: self.cut_spec_by_z()
        if ((maxmag != None) | (minmag != None)): 
            if ((self.field == 'cl1604') & prioritize_acs):
                self.cut_spec_by_mag(ACSoverride=True)
            else:
                self.cut_spec_by_mag(maxmag, minmag)

    def cut_spec_by_mag(self, maxmag=40, minmag=None, acs_override=False):
        # Cuts spectroscopic catalog by outlier magnitudes
        if acs_override:
            self.specdf = self.specdf[(self.specdf.F814W < maxmag)]
            if minmag != None: 
                self.specdf = self.specdf[(self.specdf.F814W > minmag)]
            self.specdf = self.specdf[(self.specdf.F606W - self.specdf.F814W < 3)]
        else:
            self.specdf = self.specdf[(self.specdf.r < maxmag) & (self.specdf.i < maxmag) & (self.specdf.z < maxmag)]
            if minmag != None: 
                self.specdf = self.specdf[(self.specdf.r > minmag) & (self.specdf.i > minmag) & (self.specdf.z > minmag)]

    def cut_spec_by_q(self, goodQ=[3, 4]):
        # Cuts spectroscopic catalog by Q (the quality flag)
        # Good quality flags are 3 and 4. -1 is technically
        # good, too, but it designates stars, which are not
        # applicable to a CMD
        self.specdf = self.specdf[np.in1d(self.specdf.Q, goodQ)]

    def cut_spec_by_z(self, zlow_bound = None, zupper_bound = None):
        # Cuts spectroscopic catalog by redshift, between zlow_bound
        # and zupper_bound. Uses bounds from spec_dict if they aren't provided
        if zlow_bound == None: 
            zlow_bound = self.spec_dict[self.field]['z'][1]
        if zupper_bound == None: 
            zupper_bound = self.spec_dict[self.field]['z'][2]
        self.specdf = self.specdf[(self.specdf.redshift >= zlow_bound) & (self.specdf.redshift <= zupper_bound)]
    
    def set_radec_col(self, source): #called by load_spec
        try:
            self.specdf['RA']  = self.specdf['{}_RA'.format(source)]
            self.specdf['Dec'] = self.specdf['{}_Dec'.format(source)]
        except KeyError:
            self.specdf['RA']  = self.specdf['{}_RA'.format(source)],
            self.specdf['Dec'] = self.specdf['{}_DEC'.format(source)]

    def load_phot(self, photpath='/home/rumbaugh/Chandra/photcats/', nameoverride=None, photdict={'names': ('ID', 'dum1', 'dum2', 'RA', 'Dec', 'r', 'r_err', 'i', 'i_err', 'z', 'z_err'), 'formats': ('|S64', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')}):
        # Loads photometric data file
        if nameoverride != None:
            self.photcat = '{}{}'.format(photpath, nameoverride)
        else:
            self.photcat = '{}{}_rizdata.corr.gz'.format(photpath, self.field)
        photnames = np.array(photdict['names'])
        cols2use = self.set_cols2use(len(photnames))
        self.photdf = pd.read_csv(self.photcat, names=photnames, delim_whitespace=True, usecols=cols2use)        

    def set_cols2use(self, n):
        #Creates tuple with ascending integers for use in loading files
        cols2use = (0,)
        for i in range(1,n): 
            cols2use = cols2use + (i,)
        return cols2use

    def match_spec2phot(self, match_rad=0.5/3600, merge=True):
        # Matches spectroscopic and photometric catalogs
        # using pydl.pydlutils.spheregroup.spherematch
        try:
            self.specdf, self.photdf
        except AttributeError:
            print "Must set specdf and photdf"
            return
        self.match_inds = pydl.pydlutils.spheregroup.spherematch(self.photdf.RA.values, self.photdf.Dec.values, self.specdf.RA.values, self.specdf.Dec.values, match_rad)
        matched_phot = self.photdf.iloc[self.match_inds[0]]
        matched_phot.reset_index(inplace=True)
        self.orig_specdf, self.specdf = self.specdf, self.specdf.iloc[self.match_inds[1]]
        self.specdf.reset_index(inplace = True)
        if merge:
            self.specdf = pd.merge(self.specdf, matched_phot, left_index=True, right_index=True, suffixes=('', '_phot'))
        else:
            return matched_phot
        
    def set_cmd_mags(self, use_supercolors=True, z_threshold=1.0, acs_override=True, cut_outliers = False):
        # Sets the magnitudes to be used for making CMDs
        # Options are either supercolors, or r/i/z, depending
        # on the redshift threshold set by z_threshold
        try:
            self.specdf.r_err, self.specdf.i_err, self.specdf.z_err
            specerrs = True
            if acs_override: 
                specerrs = False
        except AttributeError:
            specerrs = False
        if ((self.field == 'cl1604') & acs_override):
            self.m_b = self.specdf.F606W
            self.m_r = self.specdf.F814W
            self.m_b_name = 'F606W'
            self.m_r_name = 'F814W'
        elif use_supercolors:
            if specerrs:
                self.m_b, self.m_r, self.m_b_err, self.m_r_err = calc_supercolor_mags(self.specdf.r, self.specdf.i, self.specdf.z, self.specdf.redshift, self.specdf.r_err, self.specdf.i_err, self.specdf.z_err)
            else:
                self.m_b, self.m_r = calc_supercolor_mags(self.specdf.r, self.specdf.i, self.specdf.z, self.specdf.redshift, calc_error=False)
            self.m_b_name = r'$M_{blue}$'
            self.m_r_name = r'$M_{red}$'
        else:
            if self.spec_dict[field]['z'][0] > z_threshold:
                self.m_b = self.specdf.i
                self.m_r = self.specdf.z
                if specerrs: 
                    self.m_b_err = self.specdf.i_err
                    self.m_r_err = self.specdf.z_err 
                self.m_b_name = 'i'
                self.m_r_name = 'z'
            else:
                self.m_b = self.specdf.r
                self.m_r = self.specdf.i
                if specerrs: 
                    self.m_b_err = self.specdf.r_err
                    self.m_r_err = self.specdf.i_err 
                self.m_b_name = 'r'
                self.m_r_name = 'i'
        if cut_outliers:
            cut_inds = np.arange(len(self.m_b))[(self.m_b < 40) & (self.m_r < 40) & (self.m_b - self.m_r < 4)]
            self.m_b = self.m_b[cut_inds]
            self.m_r = self.m_r[cut_inds]
            self.specdf = self.specdf.iloc[cut_inds]
            if specerrs: 
                self.m_b_err = self.m_b_err[cut_inds]
                self.m_r_err = self.m_r_err [cut_inds]

    def plot_cmd(self, show=True, figure=1, clear=True, xmax=None, xmin=None, ymax=None, ymin=None, plotoverride=False):
        # Plots a color-magnitude diagram
        try:
            self.m_b, self.m_r
        except AttributeError:
            print "Must set m_b and m_r"
            return
        if xmax == None:
            try:
                xmax = self.xmax
            except:
                xmax = None
        if xmin == None:
            try:
                xmin = self.xmin
            except:
                xmin = None
        if ymax == None:
            try:
                ymax = self.ymax
            except:
                ymax = None
        if ymin == None:
            try:
                ymin = self.ymin
            except:
                ymin = None
        try:
            self.ax
        except AttributeError:
            plotoverride = True
        if plotoverride:
            self.fig = plt.figure(figure)
            self.ax = self.fig.add_subplot(111)
        if clear: 
            self.ax.cla()
        set_plot_params()
        self.ax.scatter(self.m_r, self.m_b - self.m_r)
        self.ax.set_xlabel(self.m_r_name)
        self.ax.set_ylabel('{} - {}'.format(self.m_b_name, self.m_r_name))
        self.ax.set_title(self.field.upper())
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if xmax != None:
            self.xmax = xmax
            if (xmax < xlim[1]): 
                xlim = (xlim[0], xmax)
        if (xmin != None):
            self.xmin = xmin
            if (xmin > xlim[0]):
                xlim = (xmin, xlim[1])
        if (ymax != None):
            self.ymax = ymax
            if (ymax < ylim[1]):
                ylim = (ylim[0], ymax)
        if (ymin != None):
            self.ymin = ymin
            if (ymin > ylim[0]): 
                ylim = (ymin, ylim[1])
        self.ax.set_xlim(xlim[0], xlim[1])
        self.ax.set_ylim(ylim[0], ylim[1])
        if show: 
            plt.show(block = False)

    def plot_rso(self, plot_rs_bounds=True, rsostep=0.25, show=True, figure=2, clear=True):
        # Plots histogram of red sequence offsets
        try:
            self.m_b, self.m_r, self.y0, self.m, self.wid
        except AttributeError:
            print "Must set magnitudes and RS parameters"
            return
        try:
            self.rsoax
        except AttributeError:
            self.rsofig = plt.figure(figure)
            self.rsoax = self.rsofig.add_subplot(111)
        if clear: 
            self.rsoax.cla()
        set_plot_params()
        rsos = (RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)) / (self.wid * 0.5)
        # Creates a nice range for plotting, using bin widths 
        # equal to rsostep
        rangelow_bound   = np.floor(np.min(rsos)/rsostep)*rsostep
        rangeupper_bound = np.ceil(np.max(rsos)/rsostep)*rsostep
        nbins = int((rangeupper_bound - rangelow_bound)/rsostep)
        # Plot histogram of rsos
        self.rsoax.hist(rsos, bins=nbins, range=(rangelow_bound, rangeupper_bound))
        # Plot bounds of red sequence
        if plot_rs_bounds:
            self.rsoax.axvline(-1, ls='dashed', lw=2, color='k')
            self.rsoax.axvline(+1, ls='dashed', lw=2, color='k')
        self.rsoax.set_xlabel('Red Sequence Offset')
        self.rsoax.set_ylabel('Number of galaxies')
        self.rsoax.set_title(self.field.upper())
        plt.show(block=False)
        
    def set_rs_init(self, figure=1, num_sigma=None, point=None, slope=None, width=None, plotcmd=True):
        # Uses user input to initialize RS parameters/fitting region
        try:
            self.ax
        except AttributeError:
            print "Plot CMD first"
            return
        if num_sigma == None:
            self.num_sigma = 3
            if self.field in ['cl1324', 'cl1604']: 
               self.num_sigma = 2
        else:
            self.num_sigma = num_sigma
        if ((point != None) & (slope != None) & (width != None)):
            x, y, m, wid = point[0], point[1], slope, width
            self.y0, self.m, self.wid, self.sig = m*(-x) + y, m, wid, wid/(2.*self.num_sigma)
            if plotcmd: 
                self.plot_cmd()
            plot_RS(self.ax, self.y0, self.m, self.wid, plotcenter = True)
            return
        elif np.count_nonzero(np.array([point != None, slope != None, width != None])) > 0:
            print "All of point, slope, and width must be set to bypass manual setting."
        x, y, m, wid = RSinit(self.ax) #Initial guess
        changeflag = False
        while not(changeflag):
            #Tweak parameters until you're satisfied
            self.plot_cmd()
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
                    self.m = m
                    self.wid = wid
                    self.sig = wid/(2.*self.num_sigma)
            else:
                print "Invalid input: %s"%inp_c
            plt.show(block=False)

    def rs_sigclip(self, stepwait=0, plotcmd=True):
        #Does sigma-clipping/linear fit algorithm to fit RS
        try:
            self.y0, self.m, self.wid, self.sig
        except AttributeError:
            print "Run set_rs_init first"
            return
        # Do initial linear fit to RS model
        onRS = np.abs(RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)) <= 0.5*self.wid
        rsos = RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)
        self.m, self.y0, rvalue, pvalue, stderr = linregress(self.m_r[onRS], self.m_b[onRS] - self.m_r[onRS])
        # Do 1-D sigma clip in color space
        rsos = RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)
        onRS = np.abs(rsos) <= 0.5*self.wid
        prevonRS = np.copy(onRS)
        converged = False
        for iteration in range(0,100):
            onRS = np.abs(rsos)  <= np.std(rsos[onRS]) * self.num_sigma
            # End when there is no change in RS membership
            if np.count_nonzero(onRS != prevonRS) > 0: 
                converged = True
                continue
            if stepwait > 0:
                # Optional step-by-step plotting
                print iteration
                self.plot_cmd()
                plot_RS(self.ax, self.y0, self.m, self.num_sigma*2*np.std(rsos[onRS]), plotcenter = False)
                time.sleep(stepwait)
            prevonRS = np.copy(onRS)
        if not(converged): 
            print "rs_sigclip failed to converge"
        self.sig = np.std(rsos[onRS])
        self.wid = 2 * self.num_sigma * self.sig
        if plotcmd: 
            self.plot_cmd()
        plot_RS(self.ax, self.y0, self.m, self.wid, plotcenter=False)
        plt.show(block=False)

    def rs_model(self, stepwait=0):
        #Does sigma-clipping/linear fit algorithm to fit RS
        try:
            self.y0, self.m, self.wid, self.sig
        except AttributeError:
            print "Run set_rs_init first"
            return
        dofit = True
        onRS = np.abs(RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)) <= self.wid
        prevonRS = np.copy(onRS)
        while dofit:
            self.m, self.y0, rvalue, pvalue, stderr = linregress(self.m_r[onRS], self.m_b[onRS] - self.m_r[onRS])
            onRS = np.abs(RS_model(self.m_r, self.y0, self.m) - (self.m_b - self.m_r)) <= self.wid
            self.sig = np.std(RS_model(self.m_r[onRS], self.y0, self.m) - (self.m_b[onRS] - self.m_r[onRS]))
            self.wid = (2.*self.num_sigma)*self.sig
            onRS = np.abs(RS_model(self.m_r, self.y0, self.m) - (self.m_b -self.m_r)) <= self.wid
            # End when there is no change in RS membership
            if np.count_nonzero(onRS != prevonRS) > 0: 
                dofit = False 
            if stepwait > 0:
                self.plot_cmd()
                plot_RS(self.ax, self.y0, self.m, self.wid, plotcenter=False)
                time.sleep(stepwait)
            prevonRS = np.copy(onRS)

    def rs_mixture(self, covar_type='full'):
        # Clustering analysis of RS and BC using GMMs
        try:
            self.m_b, self.m_r
        except AttributeError:
            print "Set magnitudes first"
            return
        X = pd.DataFrame({'m_r': self.m_r, 'color': self.m_b - self.m_r})[['m_r','color']]
        self.GMM = mixture.GaussianMixture(2, covariance_type = covar_type)
        self.GMM.fit(X)

    def plot_rs_mixture(self, plotcmd=False):
        # Plot Gaussians
        try:
            self.GMM, self.ax
        except AttributeError:
            print "run rs_mixture and created CMD plot first"
        # Predict which Gaussian correspons to RS
        if self.GMM.means_[:,1][1] == np.min(self.GMM.means_[:,1]):
            cov_index = [0,1]
        else:
            cov_index = [1,0]
        if plotcmd: 
            self.plot_cmd()
        for cov, Gmean, color in zip(self.GMM.covariances_[cov_index], self.GMM.means_[cov_index], ['red','blue']):
            v, w = linalg.eigh(cov)
            angle = 180. * np.arctan2(w[0][1], w[0][0])/ np.pi
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            for v in [v, 2*v, 3*v]:
                ell = matplotlib.patches.Ellipse(Gmean, v[0], v[1], 180. + angle, color=color, fill = False, lw = 2)
                ell.set_clip_box(self.ax.bbox)
                self.ax.add_artist(ell)
        plt.show(block=False)

    def rs_kmeans(self, n_clusters=2, init='k-means++'):
        # Clustering analysis of RS and BC using k-means
        try:
            self.m_b, self.m_r
        except AttributeError:
            print "Set magnitudes first"
            return
        # means can be initialized with specific values by
        # passing an array to init (must have dimensions
        # equal to n_clusters
        if np.shape(init) != (): 
            if len(init) != n_clusters:
                print "init must have length equal to n_clusters"
                return
        X = pd.DataFrame({'m_r': self.m_r, 'color': self.m_b - self.m_r})[['m_r','color']]
        self.kmeans = cluster.KMeans(n_clusters=n_clusters, init=init)
        self.kmeans.fit(X)

    def plot_rs_kmeans(self, meshstepperc=0.001):
        # Plot k-means decision boundary
        try:
            self.kmeans, self.ax
        except AttributeError:
            print "run rs_kmeans and created CMD plot first"
        #Create meshgrid to plot decision boundary
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_min = xlim[0]
        x_max = xlim[1]
        y_min = ylim[0]
        y_max = ylim[1]
        xx, yy = np.meshgrid(np.arange(x_min, x_max, meshstepperc * (x_max - x_min)), np.arange(y_min, y_max, meshstepperc * (y_max - y_min)))
        pred = self.kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)
        self.ax.scatter(self.kmeans.cluster_centers_[:,0], self.kmeans.cluster_centers_[:,1], marker='x', color='magenta', s=50)
        self.ax.contourf(xx, yy, pred, color='cyan', alpha=0.1)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

def set_plot_params():
    # Standard plotting parameters
    plt.rc('axes',linewidth=2)
    plt.fontsize = 14
    plt.tick_params(which='major', length=8, width=2, labelsize=14)
    plt.tick_params(which='minor', length=4, width=1.5, labelsize=14)
