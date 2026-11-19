# AbacusBGS

Clustering of DESI BGS galaxies and Abacus MAPs (two-point correlation functions with jackknife errors).

## Correlation-function pipeline

Every catalog, pair count and plot is tagged with the git commit of the code that made it. This means any plot can be traced back to exact code versions:

```
bgs_catalog.py            ──>  bgs_fsf_<B>.fits, bgs_fsf_<B>_ran<i>.fits      (on pscratch)
                                     │
map_catalog.py + map_tools.py ──> maps_<M>.fits (+ maps_all_<M>.fits)   (on pscratch)
                                   boxran_L<box>_n<n>_s<seed>.fits     (box randoms, reused across versions)
angular_weights.py        ──>  wang_<A>.npz   (angular upweights w(θ), for weighting="pip_ang")
                                     ▼
run_counts.py + tools.py  ──>  corrfunc/<label>_DD_cat-<B|M>_cnt-<C>[_ang-<A>][_q<i>].npy
                               corrfunc/<label>_DR_cat-<B|M>_cnt-<C>.npy
                                     ▼
plotting notebook         ──>  figs/<name>_cat-<B|M>_cnt-<C>_plot-<P>.png
```

Each hash is the short hash of the **last commit that touched that stage's files**:

| tag | stage | files |
|-----|-------|-------|
| `B` | BGS catalog | `bgs_catalog.py` |
| `M` | MAP catalog | `map_catalog.py`, `map_tools.py` |
| `A` | angular upweights | `angular_weights.py` |
| `C` | pair counts | `run_counts.py`, `tools.py`, `columns.py` |
| `P` | plot | `tools.py`, `provenance.py`, the notebook |

### Rules
- **Settings are hardcoded and committed.** The BGS cuts live in `bgs_catalog.py`. The MAP epoch, selection, box randoms and secondary properties live in `map_catalog.py`. The binning, jackknife regions, BGS randoms and quantile splits live in `MEASUREMENTS` in `run_counts.py`. To change something, edit the file and commit it, then rerun.
- **Uncommitted code is refused.** Every stage stops with an error if its files have uncommitted changes, so an output never carries a hash that doesn't describe it.
- **Outputs are never overwritten.** A new version gets new filenames. Rerunning an existing version is skipped. Pass `run_counts.py --force` to recompute anyway.
- **Each stage finds its inputs by itself.** `run_counts.py` loads the BGS catalog matching the current `bgs_catalog.py`, and the notebook loads the most recently written counts (`prov.count_files`), whose hashes then go into the plot filename. To reproduce an old plot, `git checkout` the hashes in its filename.
- **Push your commits.** The hashes only help if the commits exist somewhere other than one machine.

### Running it (NERSC, cosmodesi environment)
```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# 1. BGS catalog + n(z)-matched randoms
srun -N 1 -C cpu -t 01:00:00 --qos interactive --account desi python bgs_catalog.py

# 1b. angular upweights from the full target catalog (only needed for *_pipang measurements)
srun -N 1 -C cpu -t 01:00:00 --qos interactive --account desi python angular_weights.py --nthreads 128

# 1c. MAP catalog(s) + box randoms (mydesienv: needs abacusnbody). Default builds "maps" (density-matched);
#     "maps_all" keeps every MAP at the epoch. Box randoms are only built if missing (--remake-randoms forces it).
srun -N 1 -C cpu -t 01:00:00 --qos interactive --account desi python map_catalog.py
python map_catalog.py maps_all

# 2. pair counts for one measurement
python run_counts.py --list
srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi python run_counts.py bgs_smlim_s --nthreads 128
srun ... python run_counts.py maps_s --nthreads 128
```

Catalogs go to `$BGSMAPS_CATALOG_DIR`, which defaults to `/pscratch/sd/c/crmiller/catalogs`. Counts go to `corrfunc/`, which git ignores.

### Seeing which versions exist
```bash
python provenance.py current                      # hashes the current code gives each stage
python provenance.py list bgs_fsf                 # BGS catalog versions on disk
python provenance.py list maps                    # MAP catalog versions on disk (also maps_all, boxran)
python provenance.py list wang                    # angular-weight versions on disk
python provenance.py list corrfunc/bgs_smlim_s    # count versions for one measurement
```
```
  hash(es)         committed         commit message                           files
  4f4459c          2026-11-16 08:23  tighten z range to 0.2-0.25              data + 1 randoms  <- current
  28576c7          2026-11-09 11:51  logMstar > 10.7 cut                      data + 1 randoms
```
Write commit messages that describe the change (e.g. "z range 0.2-0.25"), because this listing is how you'll tell versions apart. To count on an older catalog, run `run_counts.py <label> --bgs-hash <hash>` (or `--map-hash <hash>` for MAPs); for older angular weights, add `--ang-hash <hash>`.

### MAP catalogs and secondary properties
`map_catalog.py` reads the maplog files **one at a time**, so memory is set by the largest single file:
1. **Select:** read only `mult`, `node_type` and `timestep` from each file. Find the saved step nearest `TARGETZ`, then pick the MAPs to keep: the global top N by `mult` (`maps`), or all of them (`maps_all`).
2. **Build:** reopen each file with the columns the secondaries need. Compute positions, `rsdz` and every secondary property for the selected MAPs only, then drop the file.

Each MAP row has `x, y, z, vel_z, rsdz, mult`, one column per secondary property, and `FILEIDX, ROW` (where it came from in the maplogs).

**To add a secondary property:**
1. Write `func(tab, indices, **kwargs)` in `map_tools.py`. It must return one value per index. `tab` is the full table of **one** maplog file, and `indices` are the rows of the selected head nodes.
2. Add an entry to `SECONDARY` in `map_catalog.py`, e.g. `"ram_pressure": dict(func=map_tools.integratedRamPressure, load=['node_type','length','density','rel_vel'], kwargs=dict(nepochs=10))`.
3. Commit and rerun. Any secondary column can then be used as a split in `run_counts.py`, e.g. `split=("largest_merger", 4)`.

### Derived columns and splits
Quantities that only combine catalog columns, like ratios, products and logs, don't need a new catalog. Define them in `DERIVED` in `columns.py`, or write them directly in a split. `columns.py` belongs to the counts stage (`C`), so changing it never changes the catalog hash `M`.

```python
# columns.py
DERIVED = {
    "log_mult": "log10(mult)",
    "lm_mult":  "largest_merger * mult",
}
# run_counts.py MEASUREMENTS
"maps_s_lmmult": dict(MAP_DEFAULTS, mode="s", edges=S_EDGES, split=("lm_mult", 4)),               # 4 quantiles
"maps_s_mbins":  dict(MAP_DEFAULTS, mode="s", edges=S_EDGES, split=("log10(mult)", [2, 2.5, 3, 5])),  # fixed bins
```
An expression can use any catalog column, any `DERIVED` name, numpy functions by name (`log10`, `where`, `sqrt`, ...) and `np`. A `DERIVED` entry can also be a function `tab -> array`.

Use `python run_counts.py <label> --dry-run` to try a split before committing. It loads the catalog, prints each subsample's size and value range, and stops before counting. In a notebook, `columns.evaluate(maps, "lm_mult")` gives the same values a split uses.

### Plotting in a notebook
```python
import provenance as prov, tools

dd = prov.count_files("bgs_smlim_s", "DD")      # most recently written version on disk
dr = prov.count_files("bgs_smlim_s", "DR")[0]
s, s2xi0, err = tools.DavisPeebles2PCFJackknifeCoadd(dd, dr, statistic="xi0", paths=True)
plt.errorbar(s, s2xi0, yerr=err, label="BGS")
...
plt.savefig(prov.plot_name("xi0_bgs_vs_maps", dd + [dr], "bgsclustering.ipynb"))
```
- `paths=True` raises an error if the DD and DR files come from different catalog or counting versions.
- `plot_name` refuses to run if the notebook, `tools.py` or `provenance.py` have uncommitted changes.
- `count_files` uses the most recently written complete (DD + DR) version by default, and prints which one it picked. To choose a different version, pass `cat=` / `cnt=`. To require counts that match the current code, pass `current=True`.

**Notebooks:** the check compares only the notebook's code cells to the last commit. Outputs, execution counts and markdown are ignored, so running cells doesn't count as a change. It still checks the saved *file*, not the cells you actually ran. For a figure you want to keep: save the notebook, commit it, restart the kernel and rerun it, then save the figure.
