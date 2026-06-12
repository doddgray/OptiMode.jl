"""optimode — Python interface to the OptiMode.jl differentiable mode solver.

Wraps the five OptiMode.jl component packages (MaterialDispersion,
DielectricSmoothing, MaxwellEigenmodes, ModeAnalysis, ModeSweeps) via JuliaCall,
exposing the same pipeline with Python/NumPy-native interfaces:

>>> import optimode as om
>>> grid = om.Grid(4.0, 3.0, 128, 96)
>>> f_eps = om.f_eps_mats([om.Si3N4, om.SiO2])
>>> core  = om.box([0, 0], [1.6, 0.8], 1)
>>> sm    = om.smooth_eps([core], f_eps([1/1.55]), (1, 2), grid)   # (27, 2) mat data
>>> eps_inv, deps, _ = om.inv_eps_slices(sm)
>>> kmags, evecs = om.solve_k(1/1.55, eps_inv, grid, nev=1)
>>> ng = om.group_index(kmags[0], evecs[0], 1/1.55, eps_inv, deps, grid)

The first call starts an embedded Julia runtime (see ``optimode._julia`` for
configuration). Units follow OptiMode.jl: lengths/wavelengths in μm, frequencies
``ω = 1/λ`` in μm⁻¹ (c = 1), powers in W, Kerr coefficients in μm²/W.
"""

from ._julia import julia  # noqa: F401
from .grids import Grid  # noqa: F401
from . import materials as _materials
from .materials import (  # noqa: F401
    eps, index, nng, ngvd, eps_fn, kerr_n2, with_kerr_n2, rotate,
    f_eps_mats, eps_views,
)
from .smoothing import (  # noqa: F401
    box, polygon, ball, material_shape, smooth_eps, smooth_scalar, inv_eps_slices,
)
from .solvers import (  # noqa: F401
    KrylovKitEigsolve, GPUSolver, MPBSolver, solve_k, solve_omega2, k_guess,
)
from .analysis import (  # noqa: F401
    group_index, ng_gvd, E_field, rel_power_xyz, count_E_nodes, mode_viable,
    effective_area, poynting_z, mode_intensity, kerr_dielectric_perturbation,
    solve_k_kerr,
)
from .sweeps import (  # noqa: F401
    param_grid, SlurmConfig, deploy_batch, frequency_sweep, load_batch,
    batch_status, cancel_batch, gather_batch, save_summary, load_summary,
    load_fields, run_task, Batch,
)

_MATERIALS = (
    "Si3N4", "SiO2", "LiNbO3", "MgO_LiNbO3", "LiB3O5", "Si", "Ge",
    "alpha_Al2O3", "Vacuum",
)

__version__ = "0.1.0"

__all__ = [
    "julia", "Grid",
    # materials
    *_MATERIALS,
    "eps", "index", "nng", "ngvd", "eps_fn", "kerr_n2", "with_kerr_n2", "rotate",
    "f_eps_mats", "eps_views",
    # smoothing
    "box", "polygon", "ball", "material_shape", "smooth_eps", "smooth_scalar",
    "inv_eps_slices",
    # solvers
    "KrylovKitEigsolve", "GPUSolver", "MPBSolver", "solve_k", "solve_omega2",
    "k_guess",
    # analysis
    "group_index", "ng_gvd", "E_field", "rel_power_xyz", "count_E_nodes",
    "mode_viable", "effective_area", "poynting_z", "mode_intensity",
    "kerr_dielectric_perturbation", "solve_k_kerr",
    # sweeps
    "param_grid", "SlurmConfig", "deploy_batch", "frequency_sweep", "load_batch",
    "batch_status", "cancel_batch", "gather_batch", "save_summary", "load_summary",
    "load_fields", "run_task", "Batch",
]


def __getattr__(name: str):  # lazy material objects (start Julia only when needed)
    if name in _MATERIALS:
        return getattr(_materials, name)
    raise AttributeError(f"module 'optimode' has no attribute {name!r}")
