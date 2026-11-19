# MAP merger trees → Galacticus → SEDs

Goal: take dark-matter merger trees from AbacusAurora MAP logs, run the
Galacticus semi-analytic model on them, and get galaxy SEDs (first target
redshift: z ≈ 0.22).

Example simulation: `AbacusAurora_rockV2_c000_ph390_n800` (217 Mpc/h box, 800³ particles).

## Files

- `map2tree.py`: parses MAP logs into progenitor→descendant links (`parse_file`).
- `map2galacticus.py`: builds per-node arrays, validates them, and writes a Galacticus merger tree HDF5 file.
  `python map2galacticus.py <maplog.asdf> <out.hdf5> --sigma8 <value> [--no-phase-space]`

MAP logs are read with `abacusnbody.data.read_abacus.read_asdf` (abacusutils,
`aurora` branch). The unpacker is `abacusnbody/data/maplog.py` and the
layout version is `mapnode_24b`.

## MAP log format (checked against real data)

Fields per row: `pos`, `vel`, `mult`, `node_type`, `timestep`, `mult_sec`,
`pid`, `length`, plus `density`, `vel_disp`, `rel_vel`, `rel_vel_healpix` and
`lc_label`. A field that doesn't apply to a row's node type is stored as 0.

The `node_type` codes are 0 = FORMATION, 1 = MERGER, 2 = LIGHTCONE and
3 = TIMESLICE (epoch).

Trees are serialized depth-first:

```
tree  = [primary branch: latest node → … → FORMATION, with MERGER rows inline]
        + [secondary block for each merger, EARLIEST merger first]
block = same structure, recursively
```

- The primary branch starts at the latest node and runs back in time to its FORMATION node.
- Secondaries are listed in forward time order. The primary meets its MERGER rows newest first, so the k-th secondary block belongs to the k-th merger counting backward from the formation end.
- `length` is only filled on epoch nodes (2 and 3), and is 0 on FORMATION and MERGER rows. It counts the node itself plus its subtree: the rest of its branch and every secondary that merged before it. The root's `length` equals the whole tree's size.
- Trees are stored one after another. Tiling the file by root `length` was checked on the full example file: 18,672,724 rows (equal to `MapLogTotalNodes`), 225,294 trees, of which 40,946 contain mergers.
- A MERGER row's `mult` is the mass after the merger. Checked on a sample tree: post-merger `mult` equals the primary's last epoch `mult` plus the secondary's first `mult` (to within one particle).
- An epoch node at the same step as a merger records the state before the merger.
- Halos form at `Map_MinMultiplicity` = 20 particles. Smaller infalling objects show up as mass growth with no MERGER row.

## How MAP nodes become Galacticus nodes

- Halos = TIMESLICE (3) and FORMATION (0) rows. MERGER rows only mark where a secondary attaches. LIGHTCONE rows are skipped.
- Along a branch, each node's descendant is the next later node.
- A secondary branch's latest node links to the primary's first node after the merger. That node is always strictly later in time, so no two linked nodes share a time.
- `nodeIndex` = the global row number. `hostIndex` = `nodeIndex`, so there are no subhalos and Galacticus models the satellite phase itself.
- Mass = `mult × ParticleMassHMsun` (Msun/h). Redshift = `Step_to_Z[timestep]`.

## Header (`tab.meta`): always read these at runtime, never hard-code them

`Step_to_Z`, `ParticleMassHMsun`, `H0`, `Omega_M` (includes neutrinos),
`Omega_DE`, `omega_b` (physical; divide by h²), `n_s`, `BoxSizeHMpc` and
`SimName`. `TimeSliceRedshifts_Subsample` lists the timeslice redshifts. The
slice nearest z = 0.225 is z = 0.220285.

σ₈ is not in the header, so `--sigma8` is required for now. The header names
the CLASS power spectrum (`ZD_Pk_filename`, `ZD_Pk_file_redshift`), so σ₈
could be calculated from it, which would make the converter fully header-driven.

## Galacticus source

A local clone (source only, not built) is at `/Users/catherine/DESI/codes/galacticus`, commit `a9511c7` (bleeding-edge-43). Use it for class and parameter names, example parameter files (`parameters/`, `parameters/reference/`) and the test data in `testSuite/data/`. Runs happen on NERSC; keep the NERSC build at the same commit.

## Status and open items

- [x] Parser tested on one 109-row tree with 3 mergers. It has not been tested on a tree where a secondary has its own mergers. The parser raises an error with the row number if the layout breaks.
- [x] Writer and validator tested on that tree. Validation covers unique IDs, no links to missing nodes, every descendant strictly later in time, and one root per tree.
- [ ] **Run on the full file.** Check the runtime: the parser is pure-Python recursion and may need numba or vectorization.
- [x] **HDF5 names checked** against the importer (`source/merger_trees/construct/read/importer/galacticus.F90`) and the example file `testSuite/data/lightconeCrossingTestTree16284248273.hdf5`. Dataset names, `cosmology` and `units` attributes match. Added the `forestHalos` attributes: `haloMassesIncludeSubhalos` is required (Galacticus stops with an error without it), plus `treesHaveSubhalos`, `forestsAreSelfContained`, `velocitiesIncludeHubbleFlow` and `positionsArePeriodic`. Not yet tested by running Galacticus on a converted file.
- [ ] **Position and velocity units are unverified.** The code assumes comoving Mpc/h and km/s. Check the ranges of the decoded `pos` and `vel`. Is `pos` centered on 0 or running from 0 to the box size?
- [ ] σ₈: pass the value for the cosmology, or calculate it from the CLASS file.
- [ ] Mass resolution: the smallest halo is about 3.5×10¹⁰ Msun/h, so faint galaxies are missing. Consider whether Galacticus should add branches below that mass.
- [ ] Galacticus parameter file:
  - `mergerTreeConstructor` = read
  - cosmology matching the header
  - output redshift 0.220285
  - default baryonic physics to start
  - stellar populations: IMF, SSP library (e.g. FSPS), dust and nebular choices
  - SED output: an SED property extractor if the installed version has one, otherwise many narrow top-hat filters
- [ ] Test plan: run Galacticus first on its own Monte Carlo trees, then on 1 converted tree, then a few hundred, then everything.
- [ ] Analysis: check the stellar mass–halo mass relation and the stellar mass function before trusting the SEDs.

Note: the stellar population tutorial
(`galacticus/tutorials/library/06-stellar-populations.ipynb`) has not been
reviewed yet. It appears to use the Python library bindings rather than a full
model run.

## Related user code (earlier work)

The user has functions that select TIMESLICE nodes at a target redshift
(`redshiftmask`, `get_z_indices`, `compute_history_epoch`). A bug was found:
`get_z_indices` returned indices into the table *after* filtering to
`node_type == 3`, but they were applied to the unfiltered table. The fix is to
build the mask over the full table (`(node_type == 3) & (timestep == step)`).
