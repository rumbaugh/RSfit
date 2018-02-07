import numpy as np

def set_spec_dict():
    # Creates a dictionary with information on each ORELSE field
    # Provides information for opening and using spectroscopy files
    spec_dict= { \
             'cl1324': {'file': 'FINAL.cl1322.lrisplusdeimos.feb2016.nodups.cat', 'loaddict': '', 'z':[0.756, 0.65, 0.79],  'obsids': [9403, 9404, 9836, 9840]}, \
             'cl1324_north': {'file': 'FINAL.cl1322.lrisplusdeimos.feb2016.nodups.cat',  'loaddict': '', 'z':[0.756, 0.65, 0.79],  'obsids': [9403, 9840]},  \
             'cl1324_south': {'file': 'FINAL.cl1322.lrisplusdeimos.feb2016.nodups.cat',  'loaddict': '', 'z':[0.756, 0.65, 0.79],  'obsids': [9404, 9836]}, \
             'rxj1821': {'file': 'FINAL.nep5281.deimos.gioia.aug2013.nodups.cat',  'loaddict': '', 'z':[0.818, 0.8, 0.83],  'obsids': [10444, 10924]}, \
             'cl0849': {'file': 'FINAL.spectroscopic.autocompile.blemaux.SC0849.feb2017.wSimona.nodups.cat',  'loaddict': '', 'z':[1.261, 1.25, 1.28],  'obsids': [927, 1708]}, \
             'X3': {'file': 'FINAL.semifinal.spectroscopic.autocompile.blemaux.XL005.targetsonly.apr2014.cat',  'loaddict': '', 'z':[1.050, 1, 1.1], 'obsids': []}, \
             'cl0023': {'file': 'FINAL.SG0023.deimos.lris.feb2012.nodups.cat',  'loaddict': '', 'z':[0.845, 0.82, 0.87],  'obsids': [7914]}, \
             'X5': {'file': 'FINAL.spectra.Cl0023.edit.cat',  'loaddict': '', 'z':[0.845, 0.82, 0.87],  'obsids': []}, \
             'cl1604': {'file': 'FINAL.spectra.sc1604.wcompletenessmasks.feb2012.nodups.cat',  'loaddict': '', 'z':[0.900, 0.84, 0.96], 'obsids': [6932, 6933, 7343]}, \
             'cl1350': {'file': 'FINAL.spectroscopic.autocompile.blemaux.1350.dec2015.nodups.cat',  'loaddict': '', 'z':[0.804, 0.79, 0.81], 'obsids': [2229]}, \
             'X7': {'file': 'FINAL.spectroscopic.autocompile.blemaux.1429.may2015.nodups.cat', 'loaddict': '', 'z':[0.985, 0.97, 1.], 'obsids': []}, \
             'X8': {'file': 'FINAL.spectroscopic.autocompile.blemaux.N2560.apr2012.nodups.cat', 'loaddict': '', 'z':[0, 0, 0], 'obsids': []}, \
             'rcs0224': {'file': 'FINAL.spectroscopic.autocompile.blemaux.RCS0224.apr2012.nodups.cat', 'loaddict': '', 'z':[0.772, 0.76, 0.79], 'obsids': [3181,4987]}, \
             'rxj1221': {'file': 'FINAL.spectroscopic.autocompile.blemaux.RXJ1221.dec2015.nodups.cat', 'loaddict': '', 'z':[0.700, 0.69, 0.71], 'obsids': [1662]}, \
             'rxj1716': {'file': 'FINAL.spectroscopic.autocompile.blemaux.RXJ1716.may2017.nodups.cat', 'loaddict': '', 'z':[0.813, 0.8, 0.83], 'obsids': [548]}, \
             'rxj0910': {'file': 'FINAL.spectroscopic.autocompile.blemaux.sc0910.feb2016.plusT08.nodups.cat', 'loaddict': '', 'z':[1.110, 1.08, 1.15], 'obsids': [2227, 2452]}, \
             'rxj1757': {'file': 'FINAL.spectroscopic.autocompile.N200.blemaux.aug2013.nodups.cat', 'loaddict': '', 'z':[0.691, 0.68, 0.71], 'obsids': [10443, 11999]}, \
             'X10': {'file': 'spectroscopic.autocompile.blemaux.0943A.targetsonly.cat', 'loaddict': '', 'z':[0, 0, 0], 'obsids': []}, \
             'cl1137': {'file': 'FINAL.spectroscopic.autocompile.blemaux.Cl1137.mar2017.nodups.cat', 'loaddict': '', 'z':[0.959, 0.94, 0.97], 'obsids': [4161]}, \
             'rxj1053': {'file': 'FINAL.spectroscopic.autocompile.blemaux.RXJ1053.feb2016.nodups.cat', 'loaddict': '', 'z':[1.140, 1.1, 1.15], 'obsids': [4936]}}
    for field in spec_dict.keys():
        #set dictionary with column names and formats of spec file
        spec_dict[field]['specloaddict'] = {'names':('ID', 'mask', 'slit', 'ra', 'dec', 'magR', 'magI', 'magZ', 'z', 'zerr', 'Q'), 'formats':('|S16', '|S16', '|S8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i8')}
        cols2use = (0,)
        for icols in range(1, len(spec_dict[field]['specloaddict']['names'])): 
            cols2use = cols2use + (icols,)
        spec_dict[field]['coltup'] = cols2use
    #cl1604 is a special case because of ACS imaging
    spec_dict['cl1604']['specloaddict'] = {'names':('ID', 'mask', 'slit', 'ra', 'dec', 'magR', 'magI', 'magZ', 'z', 'zerr', 'Q', 'OLDIDs', 'PHOT_FLAGS', 'ACS_RA', 'ACS_DEC', 'ACS_ID', 'F606W', 'F814W'), 'formats':('|S16', '|S16', '|S8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i8', '|S24', '|S24', 'f8', 'f8', 'f8', 'f8', 'f8')}
    cols2use = (0,)
    for icols in range(1,len(spec_dict['cl1604']['specloaddict']['names'])): 
        cols2use = cols2use + (icols,)
    spec_dict['cl1604']['coltup'] = cols2use
    spec_dict['basepath'] = '/home/rumbaugh/git/ORELSE/Catalogs/Spec_z/' #path to spectroscopy files

    return spec_dict
