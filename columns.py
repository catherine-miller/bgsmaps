'''
Derived columns and splits for catalogs, written as expressions over table columns.

run_counts.py uses these for each measurement's "split" setting, and notebooks can import the same definitions:
    import columns
    columns.evaluate(maps, "largest_merger * mult")                  # -> array, one value per row
    columns.evaluate(maps, "lm_mult")                                # a DERIVED name works the same way
    masks = columns.split_masks(maps, ("lm_mult", 4))                # 4 equal-size quantiles
    masks = columns.split_masks(maps, ("log_mult", [2, 2.5, 3, 5]))  # 3 bins with these edges
    columns.add(maps, "lm_mult", "log_mult")                         # store DERIVED entries as table columns

An expression is ordinary Python with numpy arithmetic on whole columns. It can use any column of the table,
any name in DERIVED below (derived entries can use each other), and numpy functions by name (log10, sqrt,
where, ... see FUNCTIONS) or np itself. A DERIVED entry can also be a function tab -> array, for anything
awkward to write as one expression.

This file belongs to the counts stage ("cnt" in provenance.STAGE_FILES), so editing it changes the counts hash
and leaves the catalog hash alone: adding a derived column never means rebuilding a catalog. Properties that
need the maplogs (anything that walks a merger tree) belong in SECONDARY in map_catalog.py instead.
'''
import functools
import numpy as np

# ---- named derived columns: name -> expression string, or function tab -> array -------------------------
DERIVED = {
    # MAPs
    "log_mult": "log10(mult)",
    "lm_mult":  "largest_merger * mult",   # multiplicity of the largest merger (largest_merger is a fraction of mult)
}
# ----------------------------------------------------------------------------------------------------------

FUNCTIONS = {name: getattr(np, name) for name in [
    "log", "log10", "log2", "exp", "sqrt", "abs", "sign", "floor", "ceil", "round", "minimum", "maximum",
    "clip", "where", "isfinite", "hypot", "arctan2", "sin", "cos", "pi"]}
FUNCTIONS["np"] = np

@functools.lru_cache(maxsize=None)
def _compile(expr):
    try:
        return compile(expr.strip(), "<expression: {}>".format(expr), "eval")
    except SyntaxError as e:
        raise ValueError("can't parse expression '{}': {}".format(expr, e.msg)) from None

class _Namespace(dict):
    '''Looks up column / derived / function names lazily, the first time an expression uses them.'''
    def __init__(self, tab, derived):
        super().__init__()
        self.tab, self.derived, self.stack = tab, derived, []

    def __missing__(self, name):
        if name in self.tab.colnames:
            value = np.asarray(self.tab[name])
        elif name in self.derived:
            if name in self.stack:
                raise ValueError("DERIVED entries refer to each other in a loop: " + " -> ".join(self.stack + [name]))
            self.stack.append(name)
            value = self.evaluate(self.derived[name])
            self.stack.pop()
        elif name in FUNCTIONS:
            return FUNCTIONS[name]
        else:
            raise NameError("unknown name '{}'. Table columns: {}. DERIVED: {}".format(
                name, ", ".join(self.tab.colnames), ", ".join(self.derived)))
        self[name] = value
        return value

    def evaluate(self, spec):
        if callable(spec):
            value = spec(self.tab)
        else:
            try:
                value = eval(_compile(spec), {}, self)
            except NameError as e:
                raise NameError("in '{}': {}".format(spec, e)) from None
        value = np.asarray(value)
        if value.ndim == 0: value = np.full(len(self.tab), value)
        if len(value) != len(self.tab):
            raise ValueError("'{}' gave {} values for a table of {} rows".format(spec, len(value), len(self.tab)))
        return value

def evaluate(tab, expr, derived=None):
    '''Value of an expression (or a DERIVED name, or a function tab -> array) for every row of tab.'''
    return _Namespace(tab, DERIVED if derived is None else derived).evaluate(expr)

def add(tab, *names, derived=None):
    '''Add DERIVED entries (or {colname: expression} dicts) to tab as columns, in place. Returns tab.'''
    derived = DERIVED if derived is None else derived
    for n in names:
        for colname, expr in (n.items() if isinstance(n, dict) else [(n, n)]):
            tab[colname] = evaluate(tab, expr, derived)
    return tab

def nsplit(split):
    '''Number of subsamples a split makes: (expr, n) -> n quantiles, (expr, [edges]) -> len(edges) - 1 bins.'''
    if split is None: return None
    expr, spec = split
    return int(spec) if np.ndim(spec) == 0 else len(spec) - 1

def split_masks(tab, split, verbose=False, derived=None):
    '''
    Masks for split = (expr, n): n equal-size quantiles of expr, or (expr, [e0, e1, ...]): e_i <= expr < e_i+1.
    With verbose=True, prints each subsample's size and value range.
    '''
    if split is None: return None
    expr, spec = split
    values = evaluate(tab, expr, derived)
    bad = ~np.isfinite(values)
    if bad.any():
        raise ValueError("split '{}' has {} non-finite values (nan/inf)".format(expr, bad.sum()))
    n = nsplit(split)
    if np.ndim(spec) == 0:   # same ranking as tools.makeQuantileMasks
        ranks = np.argsort(np.argsort(values))
        bounds = np.arange(n + 1) * len(values) / n
        masks = [(ranks >= bounds[i]) & (ranks < bounds[i+1]) for i in range(n)]
    else:
        masks = [(values >= spec[i]) & (values < spec[i+1]) for i in range(n)]
    if verbose:
        for i, m in enumerate(masks):
            rng = "[{:.4g}, {:.4g}]".format(values[m].min(), values[m].max()) if m.any() else "(empty)"
            print("  {} q{}: {:>9} rows, values {}".format(expr, i, m.sum(), rng))
    return masks
