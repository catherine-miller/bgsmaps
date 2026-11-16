'''
Tools for selecting Abacus MAPs (from maplogs) and making box randoms.
Moved from mapclustering_highbase.ipynb. Still being workshopped: this is a module, not a pipeline stage.
The hash of the last commit touching this file tags MAP-based counts (cat-<hash>).
'''
import os
import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.table import Table, vstack
from astropy.cosmology import Planck18 as cosmo
from abacusnbody.data.read_abacus import read_asdf

# ---- merger history ----------------------------------------------------------------------------

def get_parents(maplogtab,timestep):
    parent_mask = (maplogtab['node_type'] == 3)&(maplogtab['timestep']==timestep)
    parent_indices = np.where(parent_mask)[0]
    return parent_mask, parent_indices

def get_nmergers(maplogtab,parent_indices):
    '''
    Description
        number of mergers along the branch of each parent MAP
    Arguments
        maplogtab: Astropy table of the MAP log
        parent_indices: indices of the head nodes (e.g. from get_parents)
    Returns
        array of the number of mergers each parent MAP has gone through
    '''
    def get_nmerger(index):
        length = maplogtab['length'][index] #get length of this branch
        onetree_nodetype = maplogtab['node_type'][index:index+length] #get node type of everything in branch
        #check if onetree_nodetype[-1] is always formation
        return np.count_nonzero(onetree_nodetype == 1) #merger nodes
    return np.array([get_nmerger(i) for i in parent_indices])


def largestMergerSize(maplogtab,parent_indices):
    '''
    Description
            takes in a list of head nodes and finds the largest merger for to the main branch for each
    Arguments
        maplogtab: Astropy table of MAPs
        parent_indices: indices of head nodes you're interested in (for example, all the MAPs at a certain timestep)
    Returns
        size of the largest merger to the main branch as a fraction of main branch multiplicity
    '''

    def get_largestmerger(index):
        #get length of the branch
        #check which steps in that branch are mergers
        length = maplogtab['length'][index] #get length of this branch
        onetree_nodetype = maplogtab['node_type'][index:index+length] #get node type of everything in branch
        onetree_mult = maplogtab['mult_sec'][index:index+length]
        mask = onetree_nodetype == 1 #only choose the merger nodes
        if not np.any(mask): return 0
        else:
            largestmerger_mult = np.max(onetree_mult[mask])
            return float(largestmerger_mult)/float(maplogtab['mult'][index])

    largestmerger_list = [get_largestmerger(parent_indices[i]) for i in range(len(parent_indices))]
    return np.array(largestmerger_list)

def getLastNTimeslicesBranch(maplogtab,index,Nslices):
    '''
        returns indices of the last N timeslices along a certain MAP branch
    '''
    branchlength = maplogtab[index]['length']
    timeslices = maplogtab[index:index+branchlength]['node_type'] == 3
    if np.count_nonzero(timeslices) < Nslices: Nslices = np.count_nonzero(timeslices)
    indices_local = np.where(timeslices)[0]
    return indices_local[:Nslices] + index

def integratedRamPressure(maplogtab,parent_indices,nepochs):
    '''
    Description
        takes in a list of head nodes and finds the integrated ram pressure for each
    Arguments
        maplogtab: map log table of timeslices
    '''

    def getRamPressure(index, nepochs):
        #step along the primary branch and get last n epochs of velocity dispersion and
        indices = getLastNTimeslicesBranch(maplogtab,index,nepochs)
        density = maplogtab['density'][indices]
        rel_vel = maplogtab['rel_vel'][indices]
        return np.sum(density*rel_vel**2)
    rampressure_list = [getRamPressure(parent_indices[i],nepochs) for i in range(len(parent_indices))]
    return np.array(rampressure_list)

# ---- epoch selection -----------------------------------------------------------------------

def epoch_mask(tab, targetz=0.225, node_type=3, return_z=False):
    """Boolean mask over the FULL table: nodes of `node_type` at the saved step nearest targetz."""
    step2z = np.asarray(tab.meta['Step_to_Z'])
    is_type = np.asarray(tab['node_type']) == node_type
    ts = np.asarray(tab['timestep'])

    validsteps = np.unique(ts[is_type])          # steps that actually have slices
    zs = step2z[validsteps]
    step = validsteps[np.abs(zs - targetz).argmin()]

    mask = is_type & (ts == step)
    return (mask, step2z[step]) if return_z else mask

def read_select_z(file, targetz=0.225, load=['pos','mult','node_type','timestep']):
    tab = read_asdf(file, load=load)
    mask, zsel = epoch_mask(tab, targetz, return_z=True)
    print(f'Selected z = {zsel:.4f} (target {targetz})')
    return tab[mask]

def get_z_indices(tab, targetz=0.225):
    return np.flatnonzero(epoch_mask(tab, targetz))   # indices into the full tab

def compute_history_epoch(file, targetz, functions, arguments, funcnames,
                          load=['pos','mult','node_type','timestep'], colsout=['pos','mult']):
    tab = read_asdf(file, load=load)
    indices = get_z_indices(tab, targetz)
    epochtab = tab[indices][colsout]
    for func, args, name in zip(functions, arguments, funcnames):
        epochtab[name] = func(tab, indices, *args)
    return epochtab

# ---- RSD -----------------------------------------------------------------------------------

C_KMS = const.c.to(u.km / u.s).value  # 299792.458
def getRSDpos(pos,vel,z):
    '''
    Description
        Introduces RSD to comoving positions using line-of-sight velocities. Assumes same Hubble parameter for all positions.
        Makes a linear approximation to find change in comoving position (assumes constant d_eta/dz)
    Arguments
        pos: line of sight positions in Mpc/h. Assume positive direction is moving farther away from observer.
        vel: line of sight velocities in km/s
                z: median redshift you're looking at
    '''
    beta = vel / C_KMS
    one_plus_zpec = np.sqrt((1 + beta) / (1 - beta))  # relativistic Doppler
    dz = (1 + z) * (one_plus_zpec - 1)                  # (1+z_obs) = (1+z)(1+z_pec)
    H_h = 100.0 * cosmo.efunc(z)                        # H(z)/h in km/s/(Mpc/h)
    return pos + C_KMS * dz / H_h                       # Mpc/h

# ---- catalog helpers (from the inline notebook cells) --------------------------------------

def find_maplog_files(mapdir):
    '''all .asdf files under mapdir, sorted'''
    filelist = []
    for root, dirs, files in os.walk(mapdir):
        for file in files:
            if "asdf" in file:
                filelist.append(os.path.join(root, file))
    return sorted(filelist)

def load_maps_at_z(mapdir, targetz=0.225, rsd=True):
    '''
    MAPs at the saved step nearest targetz from every maplog file in mapdir, with x, y, z columns (Mpc/h)
    and, if rsd, rsdz = z displaced along the z axis by the z velocity.
    '''
    tab = vstack([read_select_z(f, targetz, load=['pos','mult','node_type','timestep','vel'])
                  for f in find_maplog_files(mapdir)])
    tab['x'] = tab['pos'][:,0]
    tab['y'] = tab['pos'][:,1]
    tab['z'] = tab['pos'][:,2]
    tab['vel_z'] = tab['vel'][:,2]
    if rsd: tab['rsdz'] = getRSDpos(tab['z'], tab['vel_z'], targetz)
    return tab

def select_density_matched(tab, boxlength, goaldensity, column='mult'):
    '''
    The boxlength**3 * goaldensity MAPs with the largest `column`, so the sample matches a target number density.
    Prints the resulting minimum multiplicity.
    '''
    numgoal = int(boxlength**3 * goaldensity)
    order = np.argsort(np.asarray(tab[column]), kind='stable')
    selected = tab[order[-numgoal:]]
    print("selected {} MAPs, minimum {} = {}".format(numgoal, column, np.min(selected[column])))
    return selected

def box_randoms(n, boxlength, seed, rsd=True):
    '''n uniform randoms in [-boxlength/2, boxlength/2]^3. If rsd, adds rsdz = z (randoms have no velocities).'''
    rng = np.random.default_rng(seed)
    rantab = Table()
    rantab['x'], rantab['y'], rantab['z'] = rng.random((3, n))*boxlength - boxlength/2
    if rsd: rantab['rsdz'] = rantab['z']
    return rantab
