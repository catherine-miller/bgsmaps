'''
Commit-hash provenance for the BGS / Abacus MAP correlation-function pipeline.

Every pipeline output is tagged with the short hash of the last commit that touched the code
that produced it. Stages refuse to run when that code has uncommitted changes, so a hash always
identifies exactly what made a file.

    bgs_catalog.py            -> <CATALOG_DIR>/bgs_fsf_<B>.fits, bgs_fsf_<B>_ran<i>.fits
    angular_weights.py        -> <CATALOG_DIR>/wang_<A>.npz
    run_counts.py + tools.py  -> corrfunc/<label>_{DD,DR}_cat-<B|M>_cnt-<C>[_ang-<A>][_q<i>].npy
    plotting notebook         -> figs/<base>_cat-<...>_cnt-<C>_plot-<P>.png

See which versions exist on disk:
    python provenance.py list bgs_fsf
    python provenance.py list wang
    python provenance.py list corrfunc/bgs_smlim_s
'''
import os
import json
import re
import sys
import glob
import subprocess
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CATALOG_DIR = os.environ.get("BGSMAPS_CATALOG_DIR", "/pscratch/sd/c/crmiller/catalogs")
CORRFUNC_DIR = os.path.join(REPO_ROOT, "corrfunc")
FIG_DIR = os.path.join(REPO_ROOT, "figs")
HASHLEN = 7

# which source files define each stage's hash
STAGE_FILES = {
    "cat_bgs": ["bgs_catalog.py"],
    "cat_map": ["map_tools.py"],
    "cnt": ["run_counts.py", "tools.py"],
    "ang": ["angular_weights.py"],
}

class DirtyTreeError(RuntimeError):
    pass

def _git(*args):
    out = subprocess.run(["git", "-C", REPO_ROOT, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("git {} failed: {}".format(" ".join(args), out.stderr.strip()))
    return out.stdout

def _abspaths(paths):
    return [p if os.path.isabs(p) else os.path.join(REPO_ROOT, p) for p in paths]

def file_hash(paths):
    '''
    Short hash of the last commit that touched any of paths (relative to the repo root, or absolute).
    '''
    full = _git("log", "-1", "--format=%H", "--", *_abspaths(paths)).strip()
    if not full:
        raise RuntimeError("no commit touches {} -- commit these files first".format(paths))
    return full[:HASHLEN]

def _notebook_code(text):
    '''Sources of a notebook's code cells. Outputs, execution counts, markdown and metadata are ignored.'''
    cells = json.loads(text).get("cells", [])
    return [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            for c in cells if c.get("cell_type") == "code"]

def _notebook_code_changed(relpath):
    '''True if the code cells of a notebook differ from the last commit (running cells / autosave don't count).'''
    try:
        committed = _git("show", "HEAD:" + relpath)
    except RuntimeError:
        return True
    with open(os.path.join(REPO_ROOT, relpath)) as f:
        current = f.read()
    try:
        return _notebook_code(current) != _notebook_code(committed)
    except (ValueError, KeyError):
        return True

def dirty_files(paths):
    '''
    Files among paths that are untracked or have staged/unstaged changes.
    For notebooks (.ipynb), only changes to code cells count, so running cells and autosaving outputs is fine.
    '''
    abspaths = _abspaths(paths)
    dirty = []
    for p in abspaths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        tracked = subprocess.run(["git", "-C", REPO_ROOT, "ls-files", "--error-unmatch", p],
                                 capture_output=True).returncode == 0
        if not tracked:
            dirty.append(os.path.relpath(p, REPO_ROOT) + " (untracked)")
    status = _git("status", "--porcelain", "--", *abspaths)
    for line in status.splitlines():
        if line.startswith("??"): continue
        relpath = line[3:].strip().strip('"')
        if relpath.endswith(".ipynb") and not _notebook_code_changed(relpath): continue
        dirty.append(relpath + (" (code cells changed)" if relpath.endswith(".ipynb") else ""))
    return dirty

def require_clean(paths):
    '''Raise DirtyTreeError if any of paths is untracked or has uncommitted changes.'''
    dirty = dirty_files(paths)
    if dirty:
        raise DirtyTreeError("uncommitted changes in:\n  " + "\n  ".join(dirty) +
                             "\ncommit them before running, so outputs can be tagged with a commit hash.")

def stage_hash(stage):
    return file_hash(STAGE_FILES[stage])

def current_tags():
    '''
    Hashes the current checked-out code gives each stage: {'cat_bgs': B, 'cat_map': M, 'cnt': C, 'ang': A}.
    A stage whose files have never been committed (or don't exist) gets None.
    '''
    tags = {}
    for stage in STAGE_FILES:
        try:
            tags[stage] = stage_hash(stage)
        except Exception:
            tags[stage] = None
    return tags

def commit_info(h):
    '''(date, subject) of a commit, or ('?', 'unknown commit') if git doesn't know it.'''
    try:
        date, subject = _git("log", "-1", "--format=%ci%x09%s", h).strip().split("\t", 1)
        return date[:16], subject
    except Exception:
        return "?", "unknown commit"

# ---- file naming ---------------------------------------------------------------------------

def catalog_path(h, ran=None, name="bgs_fsf"):
    base = "{}_{}".format(name, h) if ran is None else "{}_{}_ran{}".format(name, h, ran)
    return os.path.join(CATALOG_DIR, base + ".fits")

def wang_path(h):
    '''angular upweights w(theta) made by angular_weights.py at commit h'''
    return os.path.join(CATALOG_DIR, "wang_{}.npz".format(h))

def load_wang(h):
    '''(sep, weight) angular upweights from angular_weights.py commit h, in the form pycorr accepts as twopoint_weights'''
    w = np.load(wang_path(h))
    return w["sep"], w["weight"]

def counts_path(label, kind, cat, cnt, quantile=None, ang=None):
    '''kind is 'DD' or 'DR'. ang: hash of the angular weights used, if any.'''
    base = "{}_{}_cat-{}_cnt-{}".format(label, kind, cat, cnt)
    if ang is not None: base += "_ang-{}".format(ang)
    if quantile is not None: base += "_q{}".format(quantile)
    return os.path.join(CORRFUNC_DIR, base + ".npy")

_COUNTS_RE = re.compile(r"_(DD|DR)_cat-([0-9a-f]+)_cnt-([0-9a-f]+)(?:_ang-([0-9a-f]+))?(?:_q(\d+))?\.npy$")

def parse_counts_path(path):
    '''{'kind', 'cat', 'cnt', 'ang', 'quantile'} from a counts filename, or None if it isn't one. ang is None if absent.'''
    m = _COUNTS_RE.search(os.path.basename(path))
    if m is None: return None
    kind, cat, cnt, ang, q = m.groups()
    return dict(kind=kind, cat=cat, cnt=cnt, ang=ang, quantile=None if q is None else int(q))

def count_files(label, kind="DD", cat=None, cnt=None, ang=None, current=False, verbose=True):
    '''
    Counts files of one kind ('DD' or 'DR') for a measurement label, sorted by quantile.

    By default, uses the most recently written version on disk: among the (cat, cnt) versions that have both
    DD and DR files, the one whose newest file is most recent. DD and DR calls therefore pick the same version.
    cat / cnt / ang: restrict to versions with these hashes (e.g. to plot an older version)
    current: restrict to the versions matching the current code (cnt = current counts hash,
             cat = current BGS or MAP catalog hash, ang = current angular-weights hash if the counts use them)
    '''
    if current:
        tags = current_tags()
        cnt = cnt or tags["cnt"]
        cats = [cat] if cat else [tags["cat_bgs"], tags["cat_map"]]
        ang = ang or tags["ang"]
    else:
        cats = [cat] if cat else None

    versions = {}
    for f in glob.glob(os.path.join(CORRFUNC_DIR, "{}_D[DR]_cat-*.npy".format(label))):
        t = parse_counts_path(f)
        if t is None: continue
        if cats is not None and t["cat"] not in cats: continue
        if cnt is not None and t["cnt"] != cnt: continue
        if ang is not None and t["ang"] is not None and t["ang"] != ang: continue
        versions.setdefault((t["cat"], t["cnt"], t["ang"]), []).append((f, t))
    complete = {v: fs for v, fs in versions.items() if {t["kind"] for f, t in fs} == {"DD", "DR"}}
    if not complete:
        raise FileNotFoundError("no complete (DD + DR) counts for '{}' with cat={} cnt={}{}. Run: python run_counts.py {}\n"
                                "Available:\n{}".format(label, cats, cnt, " (current code)" if current else "", label,
                                                        format_available(os.path.join(CORRFUNC_DIR, label))))
    newest = max(complete, key=lambda v: max(os.path.getmtime(f) for f, t in complete[v]))
    if verbose:
        date, message = commit_info(newest[1])
        angstr = "" if newest[2] is None else " ang-{}".format(newest[2])
        print("{} {}: cat-{} cnt-{}{} ({}, {})".format(label, kind, newest[0], newest[1], angstr, date, message))
    files = [(f, t) for f, t in complete[newest] if t["kind"] == kind]
    return [f for f, t in sorted(files, key=lambda ft: ft[1]["quantile"] or 0)]

def plot_name(base, count_files, notebook, ext="png"):
    '''
    Figure path tagged with the catalog and counts hashes of the counts plotted, and the hash of the plotting code.
    Refuses if tools.py, provenance.py or the notebook have uncommitted changes.
    notebook: path to the notebook (relative to the repo root, or absolute)
    '''
    plotfiles = ["tools.py", "provenance.py", notebook]
    require_clean(plotfiles)
    P = file_hash(plotfiles)
    parsed = [parse_counts_path(f) for f in count_files]
    if any(p is None for p in parsed):
        raise ValueError("not all files are pipeline counts files: {}".format(count_files))
    cats = "-".join(sorted({p["cat"] for p in parsed}))
    cnts = "-".join(sorted({p["cnt"] for p in parsed}))
    angs = sorted({p["ang"] for p in parsed if p["ang"] is not None})
    angstr = "_ang-" + "-".join(angs) if angs else ""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, "{}_cat-{}_cnt-{}{}_plot-{}.{}".format(base, cats, cnts, angstr, P, ext))

# ---- listing what's on disk ----------------------------------------------------------------

_HASH_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{%d}(?![0-9a-f])" % HASHLEN)

def available(prefix):
    '''
    Versions on disk for files starting with prefix. prefix is a catalog name (e.g. 'bgs_fsf', looked up in
    CATALOG_DIR) or a path such as 'corrfunc/bgs_smlim_s'.
    Returns a list of dicts, newest commit first: {'tags', 'date', 'message', 'files', 'current'}.
    '''
    if os.sep in prefix:
        base = prefix if os.path.isabs(prefix) else os.path.join(REPO_ROOT, prefix)
        pattern = base + "_D[DR]_cat-*.npy"
    else:
        base = os.path.join(CATALOG_DIR, prefix)
        pattern = base + "_*"
    groups = {}
    for f in glob.glob(pattern):
        name = os.path.basename(f)
        if ".tmp." in name: continue
        tags = tuple(_HASH_RE.findall(name[len(os.path.basename(base)):]))
        if tags: groups.setdefault(tags, []).append(name)
    try:
        cur = current_tags()
    except Exception:
        cur = None
    def is_current(tags):
        if cur is None: return False
        if len(tags) == 1:                                    # catalog or angular-weights file
            return tags[0] == (cur["ang"] if os.path.basename(base) == "wang" else cur["cat_bgs"])
        cat, cnt = tags[:2]                                   # counts file (tags[2] = ang, if present)
        ok = cnt == cur["cnt"] and cat in (cur["cat_bgs"], cur["cat_map"])
        return ok and (len(tags) < 3 or tags[2] == cur["ang"])
    rows = []
    for tags, files in groups.items():
        date, message = commit_info(tags[1] if len(tags) > 1 else tags[0])
        rows.append(dict(tags=tags, date=date, message=message, files=sorted(files), current=is_current(tags)))
    return sorted(rows, key=lambda r: r["date"], reverse=True)

def _describe_files(files):
    dd = sum("_DD_" in f for f in files); dr = sum("_DR_" in f for f in files)
    if dd or dr: return "{} DD + {} DR".format(dd, dr)
    if all(f.endswith(".npz") for f in files): return "angular weights"
    ran = sum("_ran" in f for f in files)
    return "data + {} randoms".format(ran) if len(files) > ran else "{} randoms only".format(ran)

def format_available(prefix):
    rows = available(prefix)
    if not rows: return "  (nothing found for '{}')".format(prefix)
    lines = []
    for r in rows:
        tagstr = " ".join(r["tags"])
        lines.append("  {:<24} {:<16}  {:<40.40} {:<16}{}".format(
            tagstr, r["date"], r["message"], _describe_files(r["files"]), "  <- current" if r["current"] else ""))
    header = "  {:<24} {:<16}  {:<40} {:<16}".format("hash(es)", "committed", "commit message", "files")
    return "\n".join([header] + lines)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "list":
        print(format_available(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "current":
        for k, v in current_tags().items(): print("{:<8} {}".format(k, v))
    else:
        print("usage: python provenance.py list <bgs_fsf | wang | corrfunc/<label>>\n"
              "       python provenance.py current")
