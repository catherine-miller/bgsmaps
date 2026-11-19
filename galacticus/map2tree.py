"""Parse Aurora MAP logs (depth-first serialized trees) into halo-tree links.

Layout (verified on a sample tree):
  block(branch) = [branch nodes, latest -> FORMATION, MERGER nodes inline]
                  + [secondary block for each merger, earliest merger first]
  length on an epoch node = rows in its subtree (itself + rest of branch
  + secondaries that merged before it). Root length = whole tree.
"""
import numpy as np

FORMATION, MERGER, LIGHTCONE, TIMESLICE = 0, 1, 2, 3


def parse_block(s, ntype, tstep, length, out, desc_of_root, depth=0, check=True):
    """Parse the subtree starting at row s. Appends tree nodes to `out`.

    Tree nodes are TIMESLICE and FORMATION rows. MERGER rows only mark where
    secondaries attach; LIGHTCONE rows are skipped (not halo snapshots).
    Returns the number of rows consumed.
    """
    L = int(length[s])
    end = s + L
    branch_nodes = []     # tree-node rows in this branch, latest first
    mergers = []          # (merger row, row of the nearest later tree node)
    j = s
    while True:
        t = ntype[j]
        if t == MERGER:
            if not branch_nodes:
                raise ValueError(f'row {j}: merger before any node on branch')
            mergers.append((j, branch_nodes[-1]))
        elif t in (TIMESLICE, FORMATION):
            branch_nodes.append(j)
        j += 1
        if t == FORMATION:
            break
        if j >= end:
            raise ValueError(f'block at row {s}: no FORMATION before row {end}')

    # Links along the branch: each node's descendant is the previous (later) one
    for k, r in enumerate(branch_nodes):
        out.append((r, desc_of_root if k == 0 else branch_nodes[k - 1], depth))

    # Secondaries follow, earliest merger first -> mergers in reverse file order
    for m_row, desc in reversed(mergers):
        if j >= end:
            raise ValueError(f'block at row {s}: missing secondary for merger at row {m_row}')
        if check and tstep[j] > tstep[m_row]:
            raise ValueError(f'row {j}: secondary starts after its merger (row {m_row})')
        j += parse_block(j, ntype, tstep, length, out, desc, depth + 1, check)

    if j != end:
        raise ValueError(f'block at row {s}: consumed {j - s} rows, length says {L}')
    return L


def parse_file(ntype, tstep, length):
    """Return arrays (row, descendant_row, depth, tree_root_row) for all tree nodes."""
    out, roots = [], []
    i, N = 0, len(ntype)
    while i < N:
        n0 = len(out)
        i += parse_block(i, ntype, tstep, length, out, -1)
        roots += [out[n0][0]] * (len(out) - n0)
    a = np.array(out, dtype=np.int64)
    return a[:, 0], a[:, 1], a[:, 2], np.array(roots, dtype=np.int64)


if __name__ == '__main__':
    import sys
    import pandas as pd
    df = pd.read_csv(sys.argv[1])
    nt, ts, ln, mu = (df[c].to_numpy() for c in ('node_type', 'timestep', 'length', 'mult'))
    row, desc, depth, root = parse_file(nt, ts, ln)
    print(f'{len(df)} rows -> {len(row)} tree nodes, {len(np.unique(root))} trees\n')

    # Sanity: descendants are strictly later in time
    has_d = desc >= 0
    bad = has_d & (ts[desc.clip(0)] <= ts[row])
    print('non-increasing time links:', int(bad.sum()))

    # Show where branches attach and the mass budget at each merger
    i = 0
    while i < len(nt):
        L = ln[i]
        for j in range(i, i + L):
            if nt[j] == MERGER:
                after = j - 1                      # later node on primary
                before = j + 1                     # earlier node on primary
                secs = [r for r, d, dp in zip(row, desc, depth) if d == after and r != before]
                sec_m = [mu[r] for r in secs]
                print(f'merger @step {ts[j]}: mult={mu[j]}  primary before={mu[before]} (step {ts[before]})'
                      f'  secondary roots={[(int(ts[r]), int(mu[r])) for r in secs]}'
                      f'  sum={mu[before] + sum(sec_m)}')
        i += L