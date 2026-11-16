'''
Angular upweights w(theta) for BGS_BRIGHT, to correct for pairs that fiber assignment can never observe
(which PIP weights alone cannot recover). As in DESI LSS (scripts/xirunpc.py, compute_angular_weights):

    w(theta) = [parent pair counts] / [PIP-weighted fibered pair counts]     (both normalized)

parent = all targets in the full catalog; fibered = those with LOCATION_ASSIGNED, with BITWEIGHTS.
w(theta) doesn't depend on the BGS redshift / stellar-mass cuts, so it is its own stage with its own hash:
    <CATALOG_DIR>/wang_<hash>.npz          (hash = last commit touching this file)
run_counts.py applies it to DD pairs for measurements with weighting="pip_ang" and tags their counts _ang-<hash>.

Settings are hardcoded below. To change one: edit, commit, rerun.

Run on NERSC (cosmodesi environment):
    srun -N 1 -C cpu -t 01:00:00 --qos interactive --account desi python angular_weights.py --nthreads 128
'''
import os
import argparse
import numpy as np
import fitsio
from astropy.table import Table

import provenance as prov

FULL_FILE = "/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/BGS_BRIGHT_full_HPmapcut.dat.fits"
TEDGES = np.logspace(-3.5, -0.5, 31)   # degrees (0.0003 to 0.3 deg); pairs beyond get weight 1
NREALIZATIONS = 129                    # 128 fiber-assignment realizations + 1 (as for the PIP weights)

def computeAngularWeights(nthreads=None):
    '''
    Returns
        (sep, weight): pair-averaged theta in degrees, and w(theta) in each TEDGES bin
    '''
    from pycorr import TwoPointCorrelationFunction
    full = Table(fitsio.read(FULL_FILE, columns=["RA", "DEC", "LOCATION_ASSIGNED", "BITWEIGHTS"]))
    fibered = full[np.asarray(full["LOCATION_ASSIGNED"]).astype(bool)]
    print("{} parent targets, {} fibered".format(len(full), len(fibered)))
    bitweights = np.asarray(fibered["BITWEIGHTS"])
    if bitweights.ndim == 1: bitweights = bitweights[:, None]
    threadkw = {} if nthreads is None else {'nthreads': nthreads}
    wang = TwoPointCorrelationFunction('theta', TEDGES,
                                       data_positions1=[np.asarray(fibered["RA"]), np.asarray(fibered["DEC"])],
                                       data_weights1=[np.ascontiguousarray(b) for b in bitweights.T],
                                       randoms_positions1=[np.asarray(full["RA"]), np.asarray(full["DEC"])],
                                       estimator='weight', position_type='rd',
                                       weight_attrs={'normalization': 'counter', 'nrealizations': NREALIZATIONS},
                                       **threadkw)
    return np.asarray(wang.sep), np.asarray(wang.weight)

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nthreads", type=int, default=None, help="pycorr threads")
    args = parser.parse_args()

    prov.require_clean(prov.STAGE_FILES["ang"])
    h = prov.stage_hash("ang")
    path = prov.wang_path(h)
    print("angular_weights.py commit:", h)
    if os.path.exists(path):
        print("already exists, nothing to do:", path)
    else:
        os.makedirs(prov.CATALOG_DIR, exist_ok=True)
        sep, weight = computeAngularWeights(args.nthreads)
        tmp = path[:-len(".npz")] + ".tmp.npz"
        np.savez(tmp, sep=sep, weight=weight, edges=TEDGES, GITHASH=h)
        os.replace(tmp, path)
        print("wrote", path)
    sep, weight = prov.load_wang(h)
    print("{:>12}  {:>8}  {:>8}".format("theta (deg)", "w", "rp at z=0.225 (Mpc/h)"))
    for s, w in zip(sep, weight):
        print("{:12.5f}  {:8.3f}  {:8.3f}".format(s, w, np.radians(s)*638.))

if __name__ == "__main__":
    main()
