import numpy as np
from pycorr.correlation_function import JackknifeTwoPointCounter

def makeQuantileMasks(table, column, quantiles = [0,10,50,90,100]):
    '''
    Description
        sorts a table and cuts into quantiles by a certain column
    Arguments
        table: Astropy table to cut
        column: string name of column by which to sort
        quantiles: percent quantiles. Must include 0 and 100. If there are n quantiles, you'll end up with n-1 tables.
    '''
    #rank of each row in the sorted column
    ranks = np.argsort(np.argsort(table[column]))
    quantile_indices = np.asarray(quantiles)*len(table)/100
    masks = [(ranks >= quantile_indices[i]) & (ranks < quantile_indices[i+1]) for i in range(len(quantiles) - 1)]
    return masks

def makeNDivMasks(table, column, ndivs):
    '''
    Description
        sorts a table and cuts into ndivs equal-size quantiles by a certain column
    Arguments
        table: Astropy table to cut
        column: string name of column by which to sort
        ndivs: number of divisions. You'll end up with ndivs masks.
    '''
    return makeQuantileMasks(table, column, np.arange(ndivs+1)*100/ndivs)

def cutTableValues(table, column, values):
    '''
    Description
        makes masks selecting rows with values[i] <= table[column] < values[i+1]
    Arguments
        table: Astropy table to cut
        column: string name of column to cut on
        values: bin edges. If there are n values, you'll end up with n-1 masks.
    '''
    def makeMask(lowerbound, upperbound):
        return (table[column] >= lowerbound) & (table[column] < upperbound)
    masks = [makeMask(values[i], values[i+1]) for i in range(len(values)-1)]
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

def quiltMasksFromReference(reftable, tables, ndivsRA, ndivsDEC):
    '''
    Description
        like quiltMasks, but the RA/DEC boundaries are computed once from reftable and applied to every
        table, so jackknife region k covers the same patch of sky in every catalog
    Arguments
        reftable: Astropy table whose RA/DEC quantiles define the region boundaries
        tables: list of Astropy tables to split
        ndivsRA, ndivsDEC: number of divisions in RA, and in DEC within each RA strip
    Returns
        list (one entry per table) of lists of ndivsRA*ndivsDEC boolean masks
    '''
    ra_edges = np.quantile(reftable["RA"], np.arange(ndivsRA+1)/ndivsRA)
    ref_ra_bin = np.clip(np.digitize(reftable["RA"], ra_edges[1:-1]), 0, ndivsRA-1)
    dec_edges = [np.quantile(reftable["DEC"][ref_ra_bin == i], np.arange(ndivsDEC+1)/ndivsDEC) for i in range(ndivsRA)]
    def makeMasks(table):
        ra_bin = np.digitize(table["RA"], ra_edges[1:-1])
        dec_bin = np.zeros(len(ra_bin), dtype=int)
        for i in range(ndivsRA):
            in_strip = ra_bin == i
            dec_bin[in_strip] = np.digitize(table["DEC"][in_strip], dec_edges[i][1:-1])
        labels = ra_bin*ndivsDEC + dec_bin
        return [labels == k for k in range(ndivsRA*ndivsDEC)]
    return [makeMasks(table) for table in tables]

def boxMasks(tables,limits,ndivs = [3,3,3]):
    '''
    Description
        Cuts a cubic (or rectangular) mock into smaller rectangular prisms.
    Arguments
        tables: tables for which to produce masks
        limits: list of lists of box boundaries (in distance units such as Mpc/h) [[xmin, xmax],[ymin,ymax],[zmin,zmax]]
        ndivs: number of divisions in x, y, z
    Returns
        list (one entry per table) of lists of boolean masks
    '''
    ndivsx,ndivsy,ndivsz = ndivs
    xlim,ylim,zlim=limits
    x_edges = np.linspace(*xlim,num = ndivs[0] + 1)
    y_edges = np.linspace(*ylim,num = ndivs[1] + 1)
    z_edges = np.linspace(*zlim,num = ndivs[2] + 1)
    def makeMasks(table):
        x_bin = np.digitize(table["x"], x_edges[1:-1])
        y_bin = np.digitize(table["y"], y_edges[1:-1])
        z_bin = np.digitize(table["z"], z_edges[1:-1])
        labels = x_bin*ndivsy*ndivsz + y_bin*ndivsz + z_bin
        return [labels == k for k in range(ndivsx*ndivsy*ndivsz)]
    return [makeMasks(table) for table in tables]

    
def getpos(tab,position_type = 'rdd',mask = None):
    if position_type == 'rdd':
        dims = ['RA','DEC','dist']
    if position_type == 'xyz':
        dims = ['x','y','z']
    if mask is None:
        pos = [np.array(tab[dims[0]]),np.array(tab[dims[1]]),np.array(tab[dims[2]])]
    else:
        pos = [np.array(tab[dims[0]][mask]),np.array(tab[dims[1]][mask]),np.array(tab[dims[2]][mask])]
    return pos

def boolMasksToIntLabels(boolmasks):
    samples_int = np.array([(boolmasks[i]).astype(int)*(i+1) for i in range(len(boolmasks))])
    return np.sum(samples_int,axis = 0)

def getJackknifeCounts(reftab, sampletab, refmasks_jack, samplemasks_jack, stringname, samplemask = None, 
                       edges = [np.linspace(1,15,num=100),[-40,40]],position_type = 'rdd'):
    refpos = getpos(reftab,position_type=position_type)
    if samplemask is not None:
        samplepos = getpos(sampletab[samplemask],position_type = position_type)
    else:
        samplepos = getpos(sampletab,position_type = position_type)
    D2counts = len(samplepos[0])
    #ranpos = getpos(randomtab)
    if samplemask is not None:
        #mask the jackknife mask by the overall mask (e.g., if you are making a luminosity cut, mask the jackknife mask accordingly)
        samplemasks_jack = [samplemasks_jack[i][samplemask] for i in range(len(samplemasks_jack))]
    samples1 = boolMasksToIntLabels(refmasks_jack)
    samples2 = boolMasksToIntLabels(samplemasks_jack)
    #randoms_samples1 = boolMasksToIntLabels(randommasks_jack)
    counter = JackknifeTwoPointCounter('rppi', edges, positions1=refpos, samples1=samples1,
                                     positions2=samplepos, samples2=samples2,
                                     position_type = position_type)
    '''
    result = TwoPointCorrelationFunction('rppi', edges, data_positions1=refpos, data_samples1=samples1,
                                     data_positions2=samplepos, data_samples2=samples2,
                                     position_type = 'rdd')
    '''
    counter.save("corrfunc/BGSxquantile_"+stringname)   
    return D2counts

def subsampleCounts(D1tab, D2tab, D1masks_jack, D2masks_jack, D2masks_samples, stringnames, edges = [np.linspace(1,15,num=100),[-40,40]],position_type = 'rdd'):
    D2counts = []
    for i in range(len(D2masks_samples)): 
        D2mask = D2masks_samples[i]
        stringname = stringnames[i]
        D2count = getJackknifeCounts(D1tab,D2tab,D1masks_jack,D2masks_jack,stringname,samplemask = D2mask,
                                     edges = edges,position_type = position_type)
        D2counts.append(D2count)
    return D2counts

def wpDavisPeebles(D1D2s,D1R1):
    '''
    Description
        Davis-Peebles projected correlation function wp(rp) from co-added D1D2 counters and one D1R1 counter.
        Normalizations come from each counter's wnorm (N1*N2), so they always match the counts being summed,
        including for jackknife realizations with a region removed.
    Arguments
        D1D2s: list of pycorr counters (or realizations) to co-add
        D1R1: pycorr counter (or realization) for data x randoms
    '''
    D1D2norm = np.sum([D1D2.wcounts for D1D2 in D1D2s],axis = 0)/np.sum([D1D2.wnorm for D1D2 in D1D2s])
    D1R1norm = D1R1.wcounts/D1R1.wnorm
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        xi = np.where(D1R1norm > 0, D1D2norm/D1R1norm - 1, np.nan)
    #integrate over pi bins
    dpi = np.diff(D1R1.edges[1])
    return np.sum(xi*dpi,axis = 1)

def DavisPeebles2PCFJackknifeCoadd(D1D2names,D1R1name):
    '''
    Returns rp bin centers, rp*wp(rp), and jackknife error on rp*wp(rp). Bins with no D1R1 pairs are NaN.
    '''
    D1D2s = [JackknifeTwoPointCounter.load("corrfunc/"+D1D2names[i]+".npy") for i in range(len(D1D2names))]
    D1R1 = JackknifeTwoPointCounter.load("corrfunc/"+D1R1name+".npy")
    realizations = D1R1.realizations
    njack = len(realizations)
    edges = D1D2s[0].edges[0]
    rcenters = np.array([(edges[j]+edges[j+1])/2 for j in range(len(edges)-1)])
    DavisPeeblesEstimators = rcenters*wpDavisPeebles(D1D2s,D1R1)
    jacksamples = []
    for ii in realizations:
        xiJackknife = rcenters*wpDavisPeebles([D1D2.realization(ii) for D1D2 in D1D2s],D1R1.realization(ii))
        jacksamples.append(xiJackknife)
    #get jackknife error
    differencefromcenter = np.array([jack - DavisPeeblesEstimators for jack in jacksamples])
    SSD = np.sum(differencefromcenter**2, axis = 0)
    jackerror = np.sqrt((njack-1)/njack*SSD)
    return rcenters, DavisPeeblesEstimators, jackerror


