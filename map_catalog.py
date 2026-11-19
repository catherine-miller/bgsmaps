'''
Build the Abacus MAP catalog(s) at one epoch, with secondary properties (merger history, ram pressure, ...),
plus box randoms.

All settings are hardcoded below. To change one: edit this file (or map_tools.py), commit, rerun. Catalogs are
tagged with the hash of the last commit that touched map_catalog.py or map_tools.py:
    <CATALOG_DIR>/<name>_<hash>.fits                  one per entry in CATALOGS ("maps", "maps_all", ...)
    <CATALOG_DIR>/boxran_L<box>_n<n>_s<seed>.fits     box randoms, named by their parameters and reused by
                                                      every catalog version (they don't depend on the MAPs)

Maplog files are processed one at a time, so memory is bounded by one file (plus the small output):
    pass 1 (select): read only mult/node_type/timestep from each file, find the target epoch and choose
                     which MAPs to keep (global top-N by mult, or all of them)
    pass 2 (build):  reopen each file with the columns the secondaries need, compute positions, RSD and
                     every SECONDARY property for the selected MAPs only, then drop the file

Adding a secondary property: write func(tab, indices, **kwargs) -> array of len(indices) in map_tools.py,
where tab is the full table of ONE maplog file and indices are the rows of the selected head nodes in it.
Add an entry to SECONDARY with the columns it needs (load=...). Commit, rerun.

Run on NERSC (mydesienv, NOT cosmodesi):
    srun -N 1 -C cpu -t 01:00:00 --qos interactive --account desi python map_catalog.py            # "maps"
    python map_catalog.py maps maps_all
    python map_catalog.py --list
'''
import os
import sys
import argparse
import numpy as np
from astropy.table import Table, vstack
from abacusnbody.data.read_abacus import read_asdf

import provenance as prov
import map_tools

# ---- hardcoded inputs and settings (changing anything here = new catalog version) ------------
MAPDIR = "/global/cfs/cdirs/cosmo/staging/AbacusAurora/AbacusAurora_rockV2_c000_ph390_n800/maplogs/"
TARGETZ = 0.225
BOXLENGTH = 2*108.5                        # Mpc/h
GOALDENSITY = 0.00247278                   # (h/Mpc)^3, for selection="density"
BOX_RANDOMS = dict(n=5_000_000, seed=42)   # ~200x the density-matched sample; independent of the selection

SELECT_LOAD = ['mult', 'node_type', 'timestep']
BASE_LOAD = ['pos', 'vel', 'mult', 'node_type', 'timestep']

CATALOGS = {                               # named variants; each can override "randoms"=dict(n=..., seed=...)
    "maps":     dict(selection="density"), # top int(BOXLENGTH^3 * GOALDENSITY) MAPs by mult
    "maps_all": dict(selection="all"),     # every MAP at the target epoch
}

SECONDARY = {                              # column name -> func(tab, indices, **kwargs), columns it needs
    "nmergers":       dict(func=map_tools.get_nmergers, load=['node_type', 'length']),
    "largest_merger": dict(func=map_tools.largestMergerSize, load=['node_type', 'length', 'mult', 'mult_sec']),
    # TODO: nepochs=10 is a placeholder -- choose the number of most recent timeslices to sum over
    "ram_pressure":   dict(func=map_tools.integratedRamPressure, load=['node_type', 'length', 'density', 'rel_vel'],
                           kwargs=dict(nepochs=10)),
}

# ----------------------------------------------------------------------------------------------

def secondaryLoad():
    cols = list(BASE_LOAD)
    for spec in SECONDARY.values():
        cols += [c for c in spec["load"] if c not in cols]
    return cols

def selectEpoch(files, selection):
    '''
    Pass 1. Returns (step, zsel, rows, order): rows[i] = sorted row indices into file i of the selected MAPs, and
    order[i] = each selected row's rank in the final catalog (so the output order is the same as sorting
    the stacked epoch table by mult, as the old in-memory selection did).
    '''
    steps, zsels, epoch_rows, epoch_mult = [], [], [], []
    for i, f in enumerate(files):
        tab = read_asdf(f, load=SELECT_LOAD)
        mask, zsel = map_tools.epoch_mask(tab, TARGETZ, return_z=True)
        rows = np.flatnonzero(mask)
        steps.append(int(np.asarray(tab['timestep'])[rows[0]]) if len(rows) else None)
        zsels.append(zsel)
        epoch_rows.append(rows)
        epoch_mult.append(np.asarray(tab['mult'])[rows])
        print("  [{}/{}] {}: {} MAPs at z = {:.4f}".format(i+1, len(files), os.path.basename(f), len(rows), zsel))
        del tab
    if len(set(steps)) != 1:
        sys.exit("maplog files disagree on the step nearest z={}: {}".format(TARGETZ, sorted(set(steps), key=str)))

    fileidx = np.concatenate([np.full(len(r), i) for i, r in enumerate(epoch_rows)])
    rowidx = np.concatenate(epoch_rows)
    mult = np.concatenate(epoch_mult)
    if selection == "density":
        numgoal = int(BOXLENGTH**3 * GOALDENSITY)
        if numgoal > len(mult):
            print("WARNING: goal of {} MAPs exceeds the {} available; keeping all".format(numgoal, len(mult)))
            numgoal = len(mult)
        keep = np.argsort(mult, kind='stable')[-numgoal:]   # same as the old select_density_matched
        print("selected {} of {} MAPs, minimum mult = {}".format(numgoal, len(mult), mult[keep].min()))
    elif selection == "all":
        keep = np.arange(len(mult))
        print("keeping all {} MAPs".format(len(mult)))
    else:
        raise ValueError("unknown selection '{}'".format(selection))

    rank = np.arange(len(keep))   # position in keep = position in the final catalog
    rows, order = [], []
    for i in range(len(files)):
        sel = fileidx[keep] == i
        o = np.argsort(rowidx[keep][sel])
        rows.append(rowidx[keep][sel][o])
        order.append(rank[sel][o])
    return steps[0], zsels[0], rows, order

def buildFile(f, fi, rows, order):
    '''Pass 2 for one maplog file: positions, RSD and secondary properties of the selected rows.'''
    tab = read_asdf(f, load=secondaryLoad())
    out = Table()
    pos = np.asarray(tab['pos'])[rows]
    vel = np.asarray(tab['vel'])[rows]
    out['x'], out['y'], out['z'] = pos[:, 0], pos[:, 1], pos[:, 2]
    out['vel_z'] = vel[:, 2]
    out['rsdz'] = map_tools.getRSDpos(pos[:, 2], vel[:, 2], TARGETZ)
    out['mult'] = np.asarray(tab['mult'])[rows]
    for name, spec in SECONDARY.items():
        out[name] = spec["func"](tab, rows, **spec.get("kwargs", {}))
    out['FILEIDX'] = np.full(len(rows), fi, dtype=np.int32)
    out['ROW'] = rows.astype(np.int64)
    out['_order'] = order
    del tab
    return out

def buildCatalog(files, selection):
    print("pass 1: selecting MAPs")
    step, zsel, rows, order = selectEpoch(files, selection)
    print("pass 2: building catalog ({} secondary properties: {})".format(len(SECONDARY), ", ".join(SECONDARY)))
    parts = []
    for i, f in enumerate(files):
        if len(rows[i]) == 0: continue
        parts.append(buildFile(f, i, rows[i], order[i]))
        print("  [{}/{}] {}: {} MAPs".format(i+1, len(files), os.path.basename(f), len(rows[i])))
    tab = vstack(parts)
    tab.sort('_order')
    tab.remove_column('_order')
    return tab, step, zsel

def makeRandoms(ranpath, n, seed, remake):
    if os.path.exists(ranpath) and not remake:
        print("reusing box randoms", ranpath)
        return
    print("building {} box randoms".format(n))
    rantab = map_tools.box_randoms(n, BOXLENGTH, seed, rsd=True)
    prov.write_tagged(rantab, ranpath, None, BOXLEN=BOXLENGTH, NRAN=n, SEED=seed,
                      MAPTOOLS=prov.file_hash(["map_tools.py"]))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", default=["maps"], help="catalogs to build (see --list); default: maps")
    parser.add_argument("--list", action="store_true", help="list catalog names")
    parser.add_argument("--remake-randoms", action="store_true", help="rebuild the box randoms even if they exist")
    args = parser.parse_args()

    if args.list:
        for name, c in CATALOGS.items(): print("  {:<10} {}".format(name, c))
        return
    unknown = [n for n in args.names if n not in CATALOGS]
    if unknown: sys.exit("unknown catalog(s) {}. Choose from: {}".format(unknown, ", ".join(CATALOGS)))

    prov.require_clean(prov.STAGE_FILES["cat_map"])
    h = prov.stage_hash("cat_map")
    print("map_catalog.py + map_tools.py commit:", h)
    os.makedirs(prov.CATALOG_DIR, exist_ok=True)
    files = map_tools.find_maplog_files(MAPDIR)
    if not files: sys.exit("no maplog files in " + MAPDIR)

    for name in args.names:
        c = CATALOGS[name]
        ran = dict(BOX_RANDOMS, **c.get("randoms", {}))
        ranpath = prov.boxran_path(BOXLENGTH, ran["n"], ran["seed"])
        makeRandoms(ranpath, ran["n"], ran["seed"], args.remake_randoms)

        datapath = prov.catalog_path(h, name=name)
        if os.path.exists(datapath):
            print("catalog {} version {} already exists, nothing to do:".format(name, h))
            print(prov.format_available(name))
            continue
        print("building catalog '{}' from {} maplog files".format(name, len(files)))
        tab, step, zsel = buildCatalog(files, c["selection"])
        meta = dict(TARGETZ=TARGETZ, ZSEL=float(zsel), STEP=step, BOXLEN=BOXLENGTH, SELECT=c["selection"],
                    NFILES=len(files), RANFILE=os.path.basename(ranpath))
        if c["selection"] == "density": meta["GOALDENS"] = GOALDENSITY
        prov.write_tagged(tab, datapath, h, **meta)

if __name__ == "__main__":
    main()
