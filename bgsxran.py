#use cosmodesi
#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi python bgsxran.py

from abacusnbody.data.read_abacus import read_asdf
import numpy as np
import matplotlib.pyplot as plt
#from abacusnbody.analysis.tpcf_corrfunc import calc_power
import os
from astropy.table import vstack
from pycorr.correlation_function import TwoPointCorrelationFunction, JackknifeTwoPointCounter
from pycorr.twopoint_jackknife import JackknifeTwoPointEstimator
import matplotlib
from astropy.table import Table, vstack, hstack, join
import fitsio
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from tools import *

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.titlesize'] = 16

tabbgs = Table.read("/pscratch/sd/c/crmiller/bgs_fsf.ecsv", format='ascii.ecsv')

nran = 1
rans = []
for i in range(nran):
    path = "/pscratch/sd/c/crmiller/bgs_logMstar10.7_ran"+str(i)+".csv"
    rans.append(Table.read(path))

rantab = vstack(rans)
#rantab['dist'] = cosmo.comoving_distance(rantab['Z'])/u.Mpc

import importlib
import tools
importlib.reload(tools)

#split into quantiles by luminosity
#test our RA-DEC slicing algorithm
#jackknife regions share RA/DEC boundaries (set by bgs_21_35) across all catalogs
quiltmasks_bgs, quiltmasks_ran = tools.quiltMasksFromReference(tabbgs, [tabbgs, rantab], 7, 7)

R_masks = makeNDivMasks(tabbgs, "LOGMSTAR", 100)
D1tab = tabbgs
D2tab = tabbgs
D1masks_jack = quiltmasks_bgs
D2masks_jack = quiltmasks_bgs
D2masks_samples = R_masks
stringnames = []
for i in range(100):
    stringnames.append("logMstar_quant"+str(i))

start = np.log10(0.05)
stop = np.log10(5)
edges = [np.logspace(start,stop,num=20),[-40,40]]

stringname = "BGSxran"
reftab = tabbgs
sampletab = rantab
samplemasks_jack = quiltmasks_ran
refmasks_jack = quiltmasks_bgs
rancounts = tools.getJackknifeCounts(reftab, sampletab, refmasks_jack, samplemasks_jack, stringname,edges = edges,
                                    useD1weights = True, useD2weights = False)