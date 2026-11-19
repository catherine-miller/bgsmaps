"""Convert Aurora MAP logs to a Galacticus merger-tree HDF5 file.

Usage:
    python map2galacticus.py <maplog.asdf> <out.hdf5> [--sigma8 0.8076]

Tree nodes = TIMESLICE and FORMATION rows. Each secondary branch links to the
first primary node after its merger. hostIndex = nodeIndex (no subhalos);
Galacticus models the satellite phase itself.

HDF5 names and units were checked against Galacticus's importer source and
testSuite/data/lightconeCrossingTestTree16284248273.hdf5 (Galacticus commit a9511c7).
"""
import argparse
import numpy as np
import h5py

from map2tree import parse_file

# --- names in the output file: edit here if your Galacticus version differs ---
NAMES = dict(
    forest_group='forestIndex', forest_index='forestIndex',
    first_node='firstNode', n_nodes='numberOfNodes',
    halo_group='forestHalos', node_index='nodeIndex',
    desc_index='descendantIndex', host_index='hostIndex',
    mass='nodeMass', redshift='redshift', expansion='expansionFactor',
    position='position', velocity='velocity',
)

MPC_SI = 3.0856775814913673e22
MSUN_SI = 1.98847e30
KMS_SI = 1.0e3


def build_arrays(ntype, tstep, length, mult, meta, pos=None, vel=None):
    """Parse the logs and return a dict of per-node arrays plus forest table."""
    row, desc, depth, root = parse_file(ntype, tstep, length)

    step2z = np.asarray(meta['Step_to_Z'], dtype=np.float64)
    z = step2z[tstep[row]]
    mass = mult[row].astype(np.float64) * float(meta['ParticleMassHMsun'])  # Msun/h

    # Forests are contiguous in `row` order (parse_file walks trees in order)
    first = np.flatnonzero(np.r_[True, root[1:] != root[:-1]])
    nnode = np.diff(np.r_[first, len(row)])

    out = dict(nodeIndex=row, descendantIndex=desc, hostIndex=row.copy(),
               nodeMass=mass, redshift=z, expansionFactor=1.0 / (1.0 + z),
               depth=depth, forestIndex=root[first], firstNode=first,
               numberOfNodes=nnode)
    if pos is not None:
        out['position'] = np.asarray(pos)[row]
    if vel is not None:
        out['velocity'] = np.asarray(vel)[row]
    return out


def write_galacticus(path, d, meta, sigma8):
    h = float(meta['H0']) / 100.0
    N = NAMES
    with h5py.File(path, 'w') as f:
        f.attrs['formatVersion'] = 2

        g = f.create_group(N['forest_group'])
        g[N['forest_index']] = d['forestIndex']
        g[N['first_node']] = d['firstNode']
        g[N['n_nodes']] = d['numberOfNodes']

        g = f.create_group(N['halo_group'])
        g[N['node_index']] = d['nodeIndex']
        g[N['desc_index']] = d['descendantIndex']
        g[N['host_index']] = d['hostIndex']
        g[N['mass']] = d['nodeMass']
        g[N['redshift']] = d['redshift']
        g[N['expansion']] = d['expansionFactor']
        if 'position' in d:
            g[N['position']] = d['position']
        if 'velocity' in d:
            g[N['velocity']] = d['velocity']
        # Read by Galacticus's importer (source/merger_trees/construct/read/importer/
        # galacticus.F90). haloMassesIncludeSubhalos is REQUIRED (fatal error if absent).
        i32 = np.int32
        g.attrs['haloMassesIncludeSubhalos'] = i32(1)   # no subhalos, so masses are total
        g.attrs['treesHaveSubhalos'] = i32(0)           # hostIndex == nodeIndex
        g.attrs['forestsAreSelfContained'] = i32(1)     # each MAP tree is complete
        g.attrs['velocitiesIncludeHubbleFlow'] = i32(0) # assumes Abacus vel is peculiar (unverified)
        g.attrs['positionsArePeriodic'] = i32(1)        # periodic box

        c = f.create_group('cosmology')
        c.attrs['HubbleParam'] = h
        c.attrs['OmegaMatter'] = float(meta['Omega_M'])
        c.attrs['OmegaLambda'] = float(meta['Omega_DE'])
        c.attrs['OmegaBaryon'] = float(meta['omega_b']) / h**2
        c.attrs['powerSpectrumIndex'] = float(meta['n_s'])
        c.attrs['sigma_8'] = sigma8

        s = f.create_group('simulation')
        s.attrs['boxSize'] = float(meta['BoxSizeHMpc'])   # Mpc/h, comoving
        s.attrs['code'] = 'Abacus (Aurora MAP logs)'
        s.attrs['name'] = str(meta.get('SimName', ''))

        # Units: mass Msun/h, length comoving Mpc/h, velocity km/s
        u = f.create_group('units')
        u.attrs['massUnitsInSI'] = MSUN_SI
        u.attrs['massHubbleExponent'] = -1
        u.attrs['massScaleFactorExponent'] = 0
        u.attrs['lengthUnitsInSI'] = MPC_SI
        u.attrs['lengthHubbleExponent'] = -1
        u.attrs['lengthScaleFactorExponent'] = 1
        u.attrs['velocityUnitsInSI'] = KMS_SI
        u.attrs['velocityHubbleExponent'] = 0
        u.attrs['velocityScaleFactorExponent'] = 0


def validate(d):
    """Checks Galacticus will care about. Raises on failure."""
    idx = d['nodeIndex']
    assert len(np.unique(idx)) == len(idx), 'duplicate nodeIndex'
    has = d['descendantIndex'] >= 0
    pos = np.searchsorted(idx, d['descendantIndex'][has])   # idx is sorted (row order)
    assert np.all(idx[pos] == d['descendantIndex'][has]), 'dangling descendantIndex'
    assert np.all(d['redshift'][pos] < d['redshift'][has]), 'descendant not later in time'
    n_roots = np.add.reduceat((~has).astype(int), d['firstNode'])
    assert np.all(n_roots == 1), 'forest without exactly one root'
    print(f'OK: {len(idx)} nodes in {len(d["firstNode"])} forests; '
          f'mass range {d["nodeMass"].min():.3g}-{d["nodeMass"].max():.3g} Msun/h')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('maplog')
    ap.add_argument('out')
    ap.add_argument('--sigma8', type=float, required=True,
                    help='sigma_8 at z=0; the only cosmology value not in the header')
    ap.add_argument('--no-phase-space', action='store_true')
    a = ap.parse_args()

    from abacusnbody.data.read_abacus import read_asdf
    load = ['mult', 'node_type', 'timestep', 'length']
    if not a.no_phase_space:
        load += ['pos', 'vel']
    tab = read_asdf(a.maplog, load=load)
    col = lambda k: np.asarray(tab[k])
    d = build_arrays(col('node_type'), col('timestep'), col('length'), col('mult'),
                     tab.meta,
                     pos=None if a.no_phase_space else col('pos'),
                     vel=None if a.no_phase_space else col('vel'))
    validate(d)
    write_galacticus(a.out, d, tab.meta, a.sigma8)
    print('wrote', a.out)