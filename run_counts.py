'''
Compute jackknife pair counts (DD and DR) for one measurement, tagged with commit hashes:
    corrfunc/<label>_DD_cat-<catalog hash>_cnt-<counts hash>[_q<i>].npy
    corrfunc/<label>_DR_cat-<catalog hash>_cnt-<counts hash>.npy

catalog hash: BGS -> last commit touching bgs_catalog.py (override with --bgs-hash)
              MAPs -> last commit touching map_tools.py
counts hash:  last commit touching run_counts.py or tools.py

Measurements are defined in MEASUREMENTS below. To change one (or add one): edit, commit, rerun.

Usage (NERSC, cosmodesi environment):
    srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi python run_counts.py bgs_smlim_s
    python run_counts.py --list          # list measurement labels

    or for MAPs
    mydesienv (NOT cosmodesi)
    srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi python run_counts.py maps_s
    python run_counts.py --list          # list measurement labels
'''
import os
import sys
import argparse
import numpy as np
from astropy.table import Table, vstack

import provenance as prov
import tools

S_EDGES = np.logspace(np.log10(0.05), np.log10(50), num=20)
RPPI_EDGES = [np.logspace(np.log10(0.05), np.log10(5), num=20), [-40, 40]]

BGS_DEFAULTS = dict(
    sample="bgs",
    nran=1,               # number of bgs_fsf_<hash>_ran<i>.fits files to use
    ran_subsample=4,      # keep every ran_subsample-th random
    quilt=(7, 7),         # jackknife regions: RA divisions x DEC divisions
    split=None,           # e.g. ("LOGMSTAR", 4): DD for each of 4 equal-size quantiles of the data
)

MAP_DEFAULTS = dict(
    sample="maps",
    mapdir="/global/cfs/cdirs/cosmo/staging/AbacusAurora/AbacusAurora_rockV2_c000_ph390_n800/maplogs/",
    targetz=0.225,
    boxlength=2*108.5,    # Mpc/h
    goaldensity=0.00247278,
    rsd=True,             # redshift-space positions along z
    ran_factor=200,       # randoms per MAP
    ran_seed=42,
    boxdivs=(3, 3, 3),    # jackknife regions
    split=None,           # e.g. ("mult", 4)
)

MEASUREMENTS = {
    "bgs_smlim_s":  dict(BGS_DEFAULTS, mode="s", edges=S_EDGES),
    "bgs_smlim_wp": dict(BGS_DEFAULTS, mode="rppi", edges=RPPI_EDGES),
    "maps_s":       dict(MAP_DEFAULTS, mode="s", edges=S_EDGES),
    "maps_wp":      dict(MAP_DEFAULTS, mode="rppi", edges=RPPI_EDGES),
}

# ---- catalogs ------------------------------------------------------------------------------

def loadBGS(m, bgs_hash):
    datapath = prov.catalog_path(bgs_hash)
    ranpaths = [prov.catalog_path(bgs_hash, ran=i) for i in range(m["nran"])]
    missing = [p for p in [datapath] + ranpaths if not os.path.exists(p)]
    if missing:
        sys.exit("missing BGS catalog files for version {}:\n  {}\nrun bgs_catalog.py first. Available versions:\n{}"
                 .format(bgs_hash, "\n  ".join(missing), prov.format_available("bgs_fsf")))
    data = Table.read(datapath)
    rantab = vstack([Table.read(p) for p in ranpaths])[::m["ran_subsample"]]
    if data.meta.get('GITHASH') != bgs_hash:
        sys.exit("{} has GITHASH={} in its header, expected {}".format(datapath, data.meta.get('GITHASH'), bgs_hash))
    jack_data, jack_ran = tools.quiltMasksFromReference(data, [data, rantab], *m["quilt"])
    countkw = dict(position_type='rdd', los='midpoint', include_RSD=False)
    return data, rantab, jack_data, jack_ran, countkw, dict(DD=(True, True), DR=(True, False))

def loadMaps(m):
    import map_tools
    maps = map_tools.load_maps_at_z(m["mapdir"], m["targetz"], rsd=m["rsd"])
    maps = map_tools.select_density_matched(maps, m["boxlength"], m["goaldensity"])
    rantab = map_tools.box_randoms(m["ran_factor"]*len(maps), m["boxlength"], m["ran_seed"], rsd=m["rsd"])
    limits = [[-m["boxlength"]/2, m["boxlength"]/2]]*3
    jack_data, jack_ran = tools.boxMasks([maps, rantab], limits, list(m["boxdivs"]))
    countkw = dict(position_type='xyz', los='z', include_RSD=m["rsd"])
    return maps, rantab, jack_data, jack_ran, countkw, dict(DD=(False, False), DR=(False, False))

# ---- main ----------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", nargs="?", help="measurement label (see --list)")
    parser.add_argument("--list", action="store_true", help="list measurement labels")
    parser.add_argument("--bgs-hash", help="use this BGS catalog version instead of the one matching bgs_catalog.py")
    parser.add_argument("--nthreads", type=int, default=None, help="pycorr threads")
    parser.add_argument("--force", action="store_true", help="recompute even if the output files exist")
    args = parser.parse_args()

    if args.list or args.label is None:
        print("measurements:\n  " + "\n  ".join(MEASUREMENTS))
        return
    if args.label not in MEASUREMENTS:
        sys.exit("unknown measurement '{}'. Choose from: {}".format(args.label, ", ".join(MEASUREMENTS)))
    m = MEASUREMENTS[args.label]

    codefiles = prov.STAGE_FILES["cnt"] + (prov.STAGE_FILES["cat_map"] if m["sample"] == "maps" else [])
    prov.require_clean(codefiles)
    cnt = prov.stage_hash("cnt")
    if m["sample"] == "bgs":
        cat = args.bgs_hash or prov.stage_hash("cat_bgs")
        if not args.bgs_hash and prov.dirty_files(prov.STAGE_FILES["cat_bgs"]):
            print("WARNING: bgs_catalog.py has uncommitted changes; using the catalog from its last commit ({})".format(cat))
    else:
        cat = prov.stage_hash("cat_map")
    print("measurement {}: catalog cat-{}, counting code cnt-{}".format(args.label, cat, cnt))

    split = m["split"]
    nq = split[1] if split else None
    ddpaths = [prov.counts_path(args.label, "DD", cat, cnt, quantile=q) for q in (range(nq) if split else [None])]
    drpath = prov.counts_path(args.label, "DR", cat, cnt)
    if not args.force and all(os.path.exists(p) for p in ddpaths + [drpath]):
        print("counts already exist for this version (use --force to recompute):\n  " + "\n  ".join(ddpaths + [drpath]))
        return
    os.makedirs(prov.CORRFUNC_DIR, exist_ok=True)

    if m["sample"] == "bgs":
        data, rantab, jack_data, jack_ran, countkw, weights = loadBGS(m, cat)
    else:
        data, rantab, jack_data, jack_ran, countkw, weights = loadMaps(m)
    print("{} data, {} randoms, {} jackknife regions".format(len(data), len(rantab), len(jack_data)))

    common = dict(edges=m["edges"], mode=m["mode"], nthreads=args.nthreads, **countkw)
    tmp = lambda p: p[:-len(".npy")] + ".tmp.npy"

    print("counting DD")
    w1, w2 = weights["DD"]
    if split:
        samplemasks = tools.makeNDivMasks(data, split[0], nq)
        tools.subsampleCounts(data, data, jack_data, jack_data, samplemasks, None, outpaths=[tmp(p) for p in ddpaths],
                              useD1weights=w1, useD2weights=w2, **common)
    else:
        tools.getJackknifeCounts(data, data, jack_data, jack_data, None, outpath=tmp(ddpaths[0]),
                                 useD1weights=w1, useD2weights=w2, **common)
    print("counting DR")
    w1, w2 = weights["DR"]
    tools.getJackknifeCounts(data, rantab, jack_data, jack_ran, None, outpath=tmp(drpath),
                             useD1weights=w1, useD2weights=w2, **common)

    for p in ddpaths + [drpath]:
        os.replace(tmp(p), p)
        print("wrote", p)

if __name__ == "__main__":
    main()
