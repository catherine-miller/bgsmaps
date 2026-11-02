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

def getJackknifeCounts(reftab, sampletab, refmasks_jack, samplemasks_jack, stringname, samplemask = None, 
                       edges = [np.logspace(-2.5,0.5,num=100),[-40,40]]):
    refpos = getpos(reftab)
    if samplemask is not None:
        samplepos = getpos(sampletab[samplemask])
    else:
        samplepos = getpos(sampletab)
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
                                     position_type = 'rdd')
    '''
    result = TwoPointCorrelationFunction('rppi', edges, data_positions1=refpos, data_samples1=samples1,
                                     data_positions2=samplepos, data_samples2=samples2,
                                     position_type = 'rdd')
    '''
    counter.save("corrfunc/BGSxquantile_"+stringname)   
    return D2counts

def subsampleCounts(D1tab, D2tab, D1masks_jack, D2masks_jack, D2masks_samples, stringnames):
    D2counts = []
    for i in range(len(D2masks_samples)): 
        D2mask = D2masks_samples[i]
        stringname = stringnames[i]
        D2count = getJackknifeCounts(D1tab,D2tab,D1masks_jack,D2masks_jack,stringname,samplemask = D2mask)
        D2counts.append(D2count)
    return D2counts

def xiTimesRDavisPeebles(D1D2counts,D1R1counts,nD2,nR1,pimax,rcenters):
    xi = [(D1D2counts[j][0]*nR1/(D1R1counts[j][0]*nD2) - 1)*rcenters[j]*2*pimax for j in range(len(D1D2counts))]
    for i in range(len(xi)):
        if xi[i] > 1e10:
            xi[i] = 0
    return np.nan_to_num(np.array(xi),nan = 0)

def xiDavisPeebles(D1D2counts,D1R1counts,nD2,nR1,pimax):
    xi = [(D1D2counts[j][0]*nR1/(D1R1counts[j][0]*nD2) - 1)*2*pimax for j in range(len(D1D2counts))]
    return np.array(xi)

def DavisPeebles2PCFJackknifeCoadd(D1D2names,D1R1name,nD2,nR1,pimax = 40):
    D1D2s = [JackknifeTwoPointCounter.load("corrfunc/"+D1D2names[i]+".npy") for i in range(len(D1D2names))]
    D1R1 = JackknifeTwoPointCounter.load("corrfunc/"+D1R1name+".npy")
    njack = D1R1.nrealizations
    edges = D1D2s[0].edges[0]
    rcenters = np.array([(edges[j]+edges[j+1])/2 for j in range(len(edges)-1)])
    #get overall counts for estimator
    D1D2scounts = np.array([D1D2s[i].wcounts for i in range(len(D1D2names))])
    D1D2counts = np.sum(D1D2scounts,axis = 0)
    D1R1counts = D1R1.wcounts
    nD2tot = np.sum(nD2)
    DavisPeeblesEstimators = xiTimesRDavisPeebles(D1D2counts,D1R1counts,nD2tot,nR1,pimax,rcenters)
    jacksamples = []
    for i in range(D1D2s[0].nrealizations):
        D1D2scounts = np.array([D1D2s[j].realization(i+1).wcounts for j in range(len(D1D2names))])
        D1D2counts = np.sum(D1D2scounts,axis = 0)
        D1R1counts = D1R1.realization(i+1).wcounts
        xiJackknife = xiTimesRDavisPeebles(D1D2counts,D1R1counts,nD2tot,nR1,pimax,rcenters)
        jacksamples.append(xiJackknife)
    #get jackknife error
    differencefromcenter = np.array([jack - DavisPeeblesEstimators for jack in jacksamples])
    SSD = np.sum(differencefromcenter**2, axis = 0)
    jackerror = np.sqrt((njack-1)/njack*SSD)
    return rcenters, DavisPeeblesEstimators, jackerror


