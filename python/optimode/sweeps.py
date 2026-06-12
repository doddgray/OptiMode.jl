"""Batched/asynchronous mode-simulation sweeps (wraps ModeSweeps.jl).

Batches are deployed from a *setup script* (a Julia file defining
``make_problem(p)``; see ModeSweeps documentation) plus a list of parameter sets,
and run as SLURM array jobs (``backend="slurm"``), local processes
(``backend="local"``) or written out for manual submission (``backend="none"``).
Greek parameter names work as Python keywords (``param_grid(ω=[…], P=[…])``).
"""

from __future__ import annotations

from ._julia import asarray, julia, kwargs_to_julia

__all__ = [
    "param_grid", "SlurmConfig",
    "deploy_batch", "frequency_sweep", "load_batch", "batch_status", "cancel_batch",
    "gather_batch", "save_summary", "load_summary", "load_fields", "run_task",
    "Batch",
]


class Batch:
    """Handle to a deployed batch (wraps ``ModeSweeps.BatchInfo``)."""

    def __init__(self, jl_obj):
        self._jl = jl_obj

    @property
    def dir(self) -> str:
        return str(self._jl.dir)

    def __len__(self) -> int:
        return int(julia().length(self._jl))

    def status(self, verbose: bool = True) -> dict:
        return batch_status(self, verbose=verbose)

    def gather(self, **kwargs) -> list:
        return gather_batch(self, **kwargs)

    def __repr__(self) -> str:
        return f"Batch(dir={self.dir!r}, n_tasks={len(self)})"


def param_grid(**kwargs) -> list:
    """Cartesian parameter grid as a list of dicts; the first keyword varies fastest.

    Scalars broadcast; e.g. ``param_grid(ω=[0.6, 0.65], w=[1.4, 1.7], T=30.0)``
    yields 4 parameter sets. Mirrors ``ModeSweeps.param_grid`` exactly (the grid is
    built on the Julia side to guarantee identical ordering).
    """
    jl = julia()
    ps = jl.seval(
        "pairs_list -> ModeSweeps.param_grid(; (Symbol(string(first(p))) => last(p) for p in pairs_list)...)"
    )(list(kwargs.items()))
    return [dict(d.items()) for d in jl._om_rows_py(ps)]


def SlurmConfig(**kwargs):
    """SLURM submission options (``time``, ``partition``, ``mem``, ``cpus_per_task``,
    ``max_concurrent``, ``julia_flags``, ``project``, ``ssh``, ``remote_dir``, …).

    Unlike the Julia default (the deploying session's active project — which under
    JuliaCall is JuliaCall's own environment), ``project`` defaults to the OptiMode
    project so spawned workers can load the packages.
    """
    from ._julia import _default_project

    if "project" not in kwargs:
        proj = _default_project()
        if proj:
            kwargs["project"] = proj
    jl = julia()
    return jl.seval("nt -> SlurmConfig(; nt...)")(kwargs_to_julia(jl, kwargs))


def _params_julia(jl, params):
    if isinstance(params, list) and params and isinstance(params[0], dict):
        return jl._om_params([list(p.items()) for p in params])
    return params  # already a Julia object


def deploy_batch(setup_file: str, params, name: str = "", dir: str = "", nev: int = 1,
                 save_fields: bool = False, solver: str = "KrylovKitEigsolve()",
                 solver_kwargs: dict | None = None, backend: str = "slurm",
                 slurm=None, submit: bool = True) -> Batch:
    """Deploy one batch: one worker task per parameter set.

    ``params`` is a list of dicts (e.g. from :func:`param_grid`). Batch state is
    persisted in the batch directory at deploy time, so it can be monitored and
    gathered from any later session with :func:`load_batch`.
    """
    jl = julia()
    kw = dict(name=name, dir=dir, nev=int(nev), save_fields=bool(save_fields),
              solver=solver, backend=jl.Symbol(backend), submit=bool(submit))
    if solver_kwargs:
        kw["solver_kwargs"] = kwargs_to_julia(jl, solver_kwargs)
    kw["slurm"] = slurm if slurm is not None else SlurmConfig()
    return Batch(jl.deploy_batch(str(setup_file), _params_julia(jl, params), **kw))


def frequency_sweep(setup_file: str, **kwargs) -> Batch:
    """Sugar for frequency sweeps (× other parameters): ``frequency_sweep(f, ω=…, w=…)``.

    The frequency keyword ``ω`` varies fastest. Other keywords as in
    :func:`deploy_batch` (``name``, ``nev``, ``backend``, ``slurm``, …).
    """
    jl = julia()
    deploy_keys = {"name", "dir", "nev", "save_fields", "solver", "solver_kwargs",
                   "backend", "slurm", "submit"}
    sweep = {k: v for k, v in kwargs.items() if k not in deploy_keys}
    dep = {k: v for k, v in kwargs.items() if k in deploy_keys}
    params = param_grid(**sweep)
    return deploy_batch(setup_file, params, **dep)


def load_batch(dirname: str) -> Batch:
    """Re-load a batch handle from its directory (works in any session)."""
    return Batch(julia().load_batch(str(dirname)))


def batch_status(batch: Batch, verbose: bool = True) -> dict:
    """Live status: dict with ``total``, ``done``, ``failed``, ``pending``,
    ``failed_tasks`` (valid at any time while the batch runs)."""
    jl = julia()
    st = jl.batch_status(batch._jl, verbose=bool(verbose))
    return {
        "total": int(st.total), "done": int(st.done), "failed": int(st.failed),
        "pending": int(st.pending),
        "failed_tasks": [int(i) for i in st.failed_tasks],
    }


def cancel_batch(batch: Batch):
    """Cancel a running SLURM batch (``scancel``)."""
    return julia().cancel_batch(batch._jl)


def gather_batch(batch: Batch, partial: bool = True, save: bool = True,
                 formats=("csv", "tsv", "json"), fetch_fields: bool = False) -> list:
    """Collect per-task summaries into a flat table: a list of dicts, one row per
    task × band (``pandas.DataFrame(rows)`` works directly).

    ``partial=True`` gathers whatever is finished so far; failed tasks yield NaN
    rows. ``save=True`` writes ``summary.{csv,tsv,json}`` into the batch directory.
    """
    jl = julia()
    fmts = jl.seval("(xs...,) -> tuple(map(Symbol, xs)...)")(*[str(f) for f in formats])
    rows = jl.gather_batch(batch._jl, partial=bool(partial), save=bool(save),
                           formats=fmts, fetch_fields=bool(fetch_fields))
    return [dict(d.items()) for d in jl._om_rows_py(rows)]


def save_summary(rows: list, basepath: str, formats=("csv", "tsv", "json")) -> list:
    """Write a gathered summary table to ``basepath.{csv,tsv,json}``; returns paths."""
    jl = julia()
    fmts = jl.seval("(xs...,) -> tuple(map(Symbol, xs)...)")(*[str(f) for f in formats])
    jrows = jl._om_params([list(r.items()) for r in rows])
    return [str(p) for p in jl.save_summary(jrows, str(basepath), formats=fmts)]


def load_summary(path: str) -> list:
    """Re-load a summary table (``.csv``/``.tsv``/``.json``) as a list of dicts."""
    jl = julia()
    rows = jl.load_summary(str(path))
    return [dict(d.items()) for d in jl._om_rows_py(rows)]


def load_fields(batch: Batch, i: int) -> dict:
    """Full mode-field data of task ``i`` (batches deployed with ``save_fields=True``).

    Returns NumPy arrays: ``omega``, ``kmags``, ``evecs`` (list), ``Es`` (list of
    ``(3, Nx, Ny)`` complex fields), ``grid_delta``, ``grid_N``.
    """
    jl = julia()
    fd = jl.load_fields(batch._jl, int(i))
    return {
        "omega": float(fd.ω),
        "kmags": asarray(fd.kmags),
        "evecs": [asarray(v) for v in fd.evecs],
        "Es": [asarray(E) for E in fd.Es],
        "grid_delta": asarray(getattr(fd, "grid_Δ")),
        "grid_N": asarray(fd.grid_N),
    }


def run_task(batchdir: str, i: int):
    """Worker entry point: run task ``i`` of the batch in ``batchdir`` in-process."""
    return julia().run_task(str(batchdir), int(i))
