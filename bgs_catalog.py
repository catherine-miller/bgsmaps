'''
Build the BGS catalog (DESI LSS clustering catalog + fastspecfit stellar masses) and n(z)-matched randoms.

All cuts are hardcoded below. To change a cut: edit this file, commit, rerun. Outputs are tagged with the
hash of the last commit that touched this file:
    <CATALOG_DIR>/bgs_fsf_<hash>.fits
    <CATALOG_DIR>/bgs_fsf_<hash>_ran<i>.fits

Run on NERSC (cosmodesi environment):
    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    srun -N 1 -C cpu -t 01:00:00 --qos interactive --account desi python bgs_catalog.py
'''
import os
import numpy as np
import fitsio
import astropy.units as u
from astropy.table import Table, vstack, hstack, join
from astropy.cosmology import Planck18 as cosmo

import provenance as prov

# ---- hardcoded inputs and cuts (changing anything here = new catalog version) --------------
LSS_DIR = "/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y3/LSS/loa-v1/LSScats/v2/nonKP/"
LSS_TRACER = "BGS_BRIGHT"
LSS_COLUMNS = ["TARGETID", "WEIGHT", "BITWEIGHTS", "RA", "DEC", "Z"]
FSF_DIR = "/dvs_ro/cfs/cdirs/desi/vac/dr2/fastspecfit/loa/v1.0/catalogs/"
FSF_FILES = [FSF_DIR + "fastspec-loa-main-bright-nside1-hp{:02}.fits".format(i) for i in range(12)]

LSS_ZMIN, LSS_ZMAX = 0.1, 0.4   # loose redshift cut on the LSS catalog before the join
LSS_SUBSAMPLE = 4               # keep every LSS_SUBSAMPLE-th LSS galaxy
ZMIN, ZMAX = 0.2, 0.25          # final redshift range
LOGMSTAR_MIN = 10.7             # final stellar-mass cut

NRAN = 1                        # number of LSS random files (each = NGC + SGC) to turn into n(z)-matched randoms
SEED = 42                       # random file i uses default_rng(SEED + i)

# ----------------------------------------------------------------------------------------------

def comovingDistance(z):
    '''comoving distance in Mpc/h (Planck18)'''
    return np.asarray(cosmo.comoving_distance(z)/u.Mpc*cosmo.h)

def readLSS():
    '''LSS clustering catalog, NGC + SGC, with the loose z cut and subsampling'''
    caps = [Table(fitsio.read(LSS_DIR + "{}_{}_clustering.dat.fits".format(LSS_TRACER, cap), columns=LSS_COLUMNS))
            for cap in ["NGC", "SGC"]]
    lss = vstack(caps)
    lss = lss[(lss['Z'] > LSS_ZMIN) & (lss['Z'] < LSS_ZMAX)]
    return lss[::LSS_SUBSAMPLE]

def makeBGSTab(file, lss):
    '''
    Description
        Joins fastspecfit columns from one fastspecfit file onto the LSS catalog (by TARGETID) and applies the cuts.
        RA, DEC and Z come from the LSS catalog (the same redshifts the randoms are matched to).
    Arguments
        file: fastspecfit FITS file
        lss: Astropy table of LSS clustering galaxies (must have TARGETID, RA, DEC, Z)
    '''
    tab2 = Table(fitsio.read(file, ext=2, columns=["TARGETID", "ABSMAG01_SDSS_R", "LOGMSTAR"]))
    tab3 = Table(fitsio.read(file, ext=3, columns=["HALPHA_EW"]))
    tabbgs = join(hstack([tab2, tab3]), lss, keys="TARGETID")
    tabbgs = tabbgs[(tabbgs['Z'] > ZMIN) & (tabbgs['Z'] < ZMAX) & (tabbgs['LOGMSTAR'] > LOGMSTAR_MIN)]
    print("  {}: {} galaxies".format(os.path.basename(file), len(tabbgs)))
    return tabbgs

def buildBGS():
    lss = readLSS()
    tabbgs = vstack([makeBGSTab(f, lss) for f in FSF_FILES])
    tabbgs['dist'] = comovingDistance(tabbgs['Z'])
    return tabbgs

def makeRanCat(data, i):
    '''
    Description
        Random catalog i (NGC + SGC) with the LSS angular distribution and redshifts resampled from the data,
        so its n(z) matches the data.
    '''
    rantab = vstack([Table(fitsio.read(LSS_DIR + "{}_{}_{}_clustering.ran.fits".format(LSS_TRACER, cap, i),
                                       columns=["RA", "DEC", "Z"])) for cap in ["NGC", "SGC"]])
    rng = np.random.default_rng(SEED + i)
    rantab['Z'] = rng.choice(np.asarray(data['Z']), size=len(rantab), replace=True)
    rantab['dist'] = comovingDistance(rantab['Z'])
    return rantab

def writeTagged(tab, path, githash, **extra):
    tab.meta['GITHASH'] = githash
    for k, v in extra.items(): tab.meta[k] = v
    tmp = path + ".tmp"
    tab.write(tmp, format='fits', overwrite=True)
    os.replace(tmp, path)
    print("wrote", path)

def main():
    prov.require_clean(["bgs_catalog.py"])
    h = prov.file_hash(["bgs_catalog.py"])
    print("bgs_catalog.py commit:", h)
    os.makedirs(prov.CATALOG_DIR, exist_ok=True)

    datapath = prov.catalog_path(h)
    ranpaths = [prov.catalog_path(h, ran=i) for i in range(NRAN)]
    if os.path.exists(datapath) and all(os.path.exists(p) for p in ranpaths):
        print("catalog version {} already exists, nothing to do:".format(h))
        print(prov.format_available("bgs_fsf"))
        return

    if os.path.exists(datapath):
        print("loading existing", datapath)
        tabbgs = Table.read(datapath)
    else:
        print("building BGS catalog")
        tabbgs = buildBGS()
        writeTagged(tabbgs, datapath, h, ZMIN=ZMIN, ZMAX=ZMAX, LMSTMIN=LOGMSTAR_MIN)

    for i, path in enumerate(ranpaths):
        if os.path.exists(path): continue
        print("building random catalog", i)
        writeTagged(makeRanCat(tabbgs, i), path, h, SEED=SEED + i)

if __name__ == "__main__":
    main()
