'''
Compute counts of BGS galaxies x randoms for projected correlation function
Makes jackknife samples in "quilt" pattern on the sky

run with, e.g.,
    srun -N 1 -C cpu -t 4:00:00 --qos interactive --account desi python BGSxBGSran.py

'''

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

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.titlesize'] = 18

def makeBGSTab(file,bgs21_35):
    '''
    Description
        Makes an Astropy table with certain columns from fastspecfit, selecting only BGS galaxies that are also in the r < -21.35 sample
    Arguments
        file: FITS file to read from
        bgs21_35: Astropy table of BGS galaxies (must have at least TARGETID)
    Returns
        Astropy table of BGS galaxies
    '''
    columns = ["TARGETID","ABSMAG01_SDSS_U","ABSMAG01_SDSS_G","ABSMAG01_SDSS_R","DN4000"]
    tab2 = Table(fitsio.read(file,ext=2,columns=columns))
    columns = ["Z","RA","DEC"]
    tab1 = Table(fitsio.read(file,ext=1,columns=columns))
    columns = ["HALPHA_EW"]
    tab3 = Table(fitsio.read(file,ext=3,columns=columns))
    tabbgs = hstack([tab1,tab2,tab3])
    tabbgs = join(tabbgs,bgs21_35,keys="TARGETID")
    return tabbgs


# split tables by quantiles

#new plan: make a bunch of MASKS rather than saving all the sliced tables

def makeQuantileMasks(table, column, quantiles = [0,10,50,90,100]):
    '''
    Description
        sorts a table and cuts into quantiles by a certain column
    Arguments
        table: Astropy table to cut
        column: string name of column by which to sort
        quantiles: percent quantiles. Must include 0 and 100. If there are n quantiles, you'll end up with n-1 tables.
    '''
    #make the quantiles
    sorted_indices = np.argsort(table[column])
    quantile_indices = quantiles/100*len(table)
    masks = [(sorted_indices >= quantile_indices[i]) and (sorted_indices < quantile_indices[i+1]) for i in range(len(quantiles) - 1)]
    return masks

def makeNDivMasks(table, column, ndivs):
    '''
    Description
        sorts a table and cuts into quantiles by a certain column
    Arguments
        table: Astropy table to cut
        column: string name of column by which to sort
        quantiles: percent quantiles. Must include 0 and 100. If there are n quantiles, you'll end up with n-1 tables.
    '''
    #make the quantiles
    argsorted_indices = np.argsort(table[column])
    sorted_indices = np.argsort(argsorted_indices)
    div_indices = np.arange(ndivs+1)/ndivs*len(table)
    masks = [(sorted_indices >= div_indices[i]) & (sorted_indices < div_indices[i+1]) for i in range(ndivs)]
    return masks

def cutTableValues(table, column, values):
    '''
    '''
    #make the quantiles
    def makeMask(lowerbound, upperbound):
        mask = (table[column] >= lowerbound) and (table[column] < upperbound)
        return mask
    masks = [makeMask(column, values[i], values[i+1]) for i in range(len(values)-1)]
    return masks
'''
def sliceTableRADEC(table, ndivsRA, ndivsDEC):
    def sliceTab(table,ndivs,columnname):
        sortedtab = table.sort(columnname)
        indices = range(ndivs+1)/ndivs*len(table)
        indices = indices.astype(int)
        sliced = [table[indices[i]:indices[i+1]] for i in range(ndivs)]
        return sliced
    slicedRA = sliceTab(table,ndivsRA,"RA")
    slicedRADEC = [sliceTab(tab,ndivsDEC,"DEC") for tab in slicedRA]
    return [tabDEC for tabRA in slicedRADEC for tabDEC in tabRA]'''

def quiltMasks(table,ndivsRA,ndivsDEC):
    ra_masks = makeNDivMasks(table,"RA",ndivsRA)
    indices = np.arange(len(table))
    ra_indices = [indices[mask] for mask in ra_masks]
    dec_masks = [makeNDivMasks(table[mask],"DEC",ndivsDEC) for mask in ra_masks]
    ra_dec_split_indices = [ra_indices[i][dec_masks[i][j]] for i in range(ndivsRA) for j in range(ndivsDEC)]
    def makeMaskFromIndices(indices):
        boolean_mask = np.zeros(len(table), dtype=bool)
        boolean_mask[indices] = True
        return boolean_mask
    masklist = [makeMaskFromIndices(ra_dec_split_indices[i]) for i in range(ndivsRA*ndivsDEC)]
    #we don't care about keeping track of where our divisions are
    return masklist
    

def getpos(tab,mask = None):
    if mask is not None:
        pos = [np.array(tab['RA'][0]),np.array(tab['DEC'][0]),np.array(tab['dist'][0])]
    else:
        pos = [np.array(tab['RA'][mask][0]),np.array(tab['DEC'][mask][0]),np.array(tab['dist'][mask][0])]
    return pos

def boolMasksToIntLabels(boolmasks):
    samples_int = np.array([(boolmasks[i]).astype(int)*(i+1) for i in range(len(boolmasks))])
    return np.sum(samples_int,axis = 0)

print("reading bgs clustering tables")
#get table of bgs galaxies
bgsdir = "/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/nonKP/"
bgsN = Table(fitsio.read(bgsdir+"BGS_BRIGHT-21.35_NGC_clustering.dat.fits",columns = ["TARGETID","WEIGHT","RA","DEC","Z"]))
bgsS = Table(fitsio.read(bgsdir+"BGS_BRIGHT-21.35_SGC_clustering.dat.fits",columns = ["TARGETID","WEIGHT","RA","DEC","Z"]))
bgs21_35 = vstack([bgsN,bgsS])

nran = 6
print("reading "+str(nran)+" random tables")
#get table of bgs randoms
bgsdir = "/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/nonKP/"
rantabs = []
for i in range(nran):
    bgsranN = Table(fitsio.read(bgsdir+"BGS_BRIGHT-21.35_NGC_"+str(i)+"_clustering.ran.fits",columns = ["RA","DEC","Z"]))
    bgsranS = Table(fitsio.read(bgsdir+"BGS_BRIGHT-21.35_SGC_"+str(i)+"_clustering.ran.fits",columns = ["RA","DEC","Z"]))
    rantabs.append(bgsranN)
    rantabs.append(bgsranS)
bgsrantab = vstack(rantabs)

print("making quilt masks for jackknife subsampling")
masks = quiltMasks(bgs21_35,7,7)
ranmasks = quiltMasks(bgsrantab,7,7)

for mask in masks:
    bgstabcut = bgs21_35[mask]
    plt.scatter(bgstabcut["RA"],bgstabcut["DEC"],s=0.03)
plt.title("Quilt Jackknife Samples")
plt.xlabel("Right Ascension (Degrees)")
plt.ylabel("Declination (Degrees)")
plt.savefig("figs/bgsquilt.png")

print("computing comoving distances")
#get comoving distances along line of sight (Planck 2018 cosmology)
bgs21_35['dist'] = cosmo.comoving_distance(bgs21_35['Z'])/u.Mpc
bgsrantab['dist'] = cosmo.comoving_distance(bgsrantab['Z'])/u.Mpc

#compute the counts
positions1 = getpos(bgs21_35)
positions2 = getpos(bgsrantab)

#ranpos = getpos(randomtab)
print("making integer sample labels")
samples1 = boolMasksToIntLabels(masks)
samples2 = boolMasksToIntLabels(ranmasks)
print("running jackknife counter")
edges = [np.logspace(-2.5,0.5,num=100),[-40,40]]
#randoms_samples1 = boolMasksToIntLabels(randommasks_jack)
counter = JackknifeTwoPointCounter('rppi', edges, positions1=positions1, samples1=samples1,
                                 positions2=positions2, samples2=samples2,
                                 position_type = 'rdd',nthreads=128)
'''
result = TwoPointCorrelationFunction('rppi', edges, data_positions1=refpos, data_samples1=samples1,
                                 data_positions2=samplepos, data_samples2=samples2,
                                 position_type = 'rdd')
'''
counter.save("corrfunc/BGS21_35xBGSran") 