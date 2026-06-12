"""Mode post-processing and Kerr corrections (wraps ModeAnalysis.jl)."""

from __future__ import annotations

import numpy as np

from ._julia import asarray, julia, to_julia_f64
from .grids import Grid

__all__ = [
    "group_index", "ng_gvd", "E_field",
    "rel_power_xyz", "count_E_nodes", "mode_viable", "effective_area",
    "poynting_z", "mode_intensity", "kerr_dielectric_perturbation", "solve_k_kerr",
]


def _f64(jl, A):
    return to_julia_f64(jl, np.asarray(A, dtype=np.float64))


def _cplx(jl, v):
    return jl.seval("v -> Vector{ComplexF64}(v)")(np.asarray(v, dtype=np.complex128))


def group_index(k: float, evec, omega: float, eps_inv, deps_dom, grid: Grid) -> float:
    """Modal group index ``n_g = ∂k/∂ω`` from a single mode solution.

    Uses the Hellmann–Feynman expression including material dispersion through the
    smoothed ``∂ε/∂ω`` field — no extra eigensolves.
    """
    jl = julia()
    return float(
        jl.group_index(float(k), _cplx(jl, evec), float(omega),
                       _f64(jl, eps_inv), _f64(jl, deps_dom), grid._jl)
    )


def ng_gvd(omega: float, k: float, evec, eps_inv, deps_dom, d2eps_dom2, grid: Grid):
    """Group index *and* group-velocity dispersion ``∂²k/∂ω²`` (one adjoint solve).

    Returns ``(ng, gvd)``; GVD is in μm (c = 1 units; multiply by λ²/(2πc²) for β₂).
    """
    jl = julia()
    res = jl.ng_gvd(float(omega), float(k), _cplx(jl, evec),
                    _f64(jl, eps_inv), _f64(jl, deps_dom), _f64(jl, d2eps_dom2), grid._jl)
    return float(res[0]), float(res[1])


def E_field(k: float, evec, eps_inv, deps_dom, grid: Grid,
            canonicalize: bool = True, normalized: bool = True) -> np.ndarray:
    """Real-space electric field ``E`` of a mode, shape ``(3, Nx, Ny)`` complex.

    ``canonicalize`` makes the largest component purely real; ``normalized``
    rescales to the dispersive energy normalization ``∫E*·(∂ε/∂ω)·E dV = 1``.
    """
    jl = julia()
    E = jl._om_Efield(float(k), _cplx(jl, evec), _f64(jl, eps_inv), _f64(jl, deps_dom),
                      grid._jl, canonicalize=canonicalize, normalized=normalized)
    return asarray(E)


def rel_power_xyz(eps, E) -> np.ndarray:
    """Relative E-field power along x/y/z (length-3, unit norm) — mode polarization."""
    jl = julia()
    Ej = jl.seval("E -> Array{ComplexF64}(E)")(np.asarray(E, dtype=np.complex128))
    return asarray(jl.E_relpower_xyz(_f64(jl, eps), Ej))


def count_E_nodes(E, eps, pol_idx: int, rel_amp_min: float = 0.1):
    """Count field-sign nodes along x and y → Hermite–Gauss-like mode order (m, n)."""
    jl = julia()
    Ej = jl.seval("E -> Array{ComplexF64}(E)")(np.asarray(E, dtype=np.complex128))
    res = jl.count_E_nodes(Ej, _f64(jl, eps), int(pol_idx), rel_amp_min=float(rel_amp_min))
    return int(res[0]), int(res[1])


def mode_viable(E, eps, pol_idx: int = 1, mode_order=(0, 0), rel_amp_min: float = 0.4) -> bool:
    """True if ``E`` is polarized along ``pol_idx`` with the given mode order."""
    jl = julia()
    Ej = jl.seval("E -> Array{ComplexF64}(E)")(np.asarray(E, dtype=np.complex128))
    mo = jl.seval("(a, b) -> (Int(a), Int(b))")(int(mode_order[0]), int(mode_order[1]))
    return bool(jl.mode_viable(Ej, _f64(jl, eps), pol_idx=int(pol_idx),
                               mode_order=mo, rel_amp_min=float(rel_amp_min)))


def effective_area(n: float, ng: float, E) -> float:
    """Effective area ``1/(n·ng·max|E⊥|²)`` for an energy-normalized field ``E`` (μm²)."""
    jl = julia()
    Ej = jl.seval("E -> Array{ComplexF64}(E)")(np.asarray(E, dtype=np.complex128))
    return float(jl.effective_area(float(n), float(ng), Ej))


def poynting_z(k: float, evec, eps_inv, grid: Grid) -> np.ndarray:
    """z-component of the time-averaged Poynting vector (arbitrary scale), (Nx, Ny)."""
    jl = julia()
    return asarray(jl.poynting_z(float(k), _cplx(jl, evec), _f64(jl, eps_inv), grid._jl))


def mode_intensity(k: float, evec, eps_inv, grid: Grid, P: float) -> np.ndarray:
    """Modal intensity ``I(x, y)`` (W/μm²) normalized so ``∑I·dV = P`` (P in W)."""
    jl = julia()
    return asarray(jl.mode_intensity(float(k), _cplx(jl, evec), _f64(jl, eps_inv),
                                     grid._jl, float(P)))


def kerr_dielectric_perturbation(I, n2_map, eps):
    """First-order Kerr perturbation: ``Δn = n₂·I`` and diagonal ``Δε = 2n₀Δn``.

    Returns ``(delta_eps, delta_n)`` as NumPy arrays.
    """
    jl = julia()
    de, dn = jl.kerr_dielectric_perturbation(_f64(jl, I), _f64(jl, n2_map), _f64(jl, eps))
    return asarray(de), asarray(dn)


def solve_k_kerr(omega: float, P: float, eps_inv, deps_dom, n2_map, grid: Grid,
                 solver=None, nev: int = 1, **kwargs) -> dict:
    """Power-corrected mode solve with first-order Kerr (n₂) perturbation.

    Each band is re-solved with the dielectric perturbation induced by its own
    intensity profile at optical power ``P`` (W; full power assumed in that mode).
    Returns a dict with NumPy values: ``kmags``, ``evecs``, ``kmags_lin``,
    ``evecs_lin``, ``dn_max``. The power-dependent effective-index shift of band
    ``b`` is ``(kmags[b] - kmags_lin[b]) / omega``.
    """
    from .solvers import KrylovKitEigsolve

    jl = julia()
    if solver is None:
        solver = KrylovKitEigsolve()
    res = jl.solve_k_kerr(float(omega), float(P), _f64(jl, eps_inv), _f64(jl, deps_dom),
                          _f64(jl, n2_map), grid._jl, solver, nev=int(nev), **kwargs)
    return {
        "kmags": asarray(res.kmags),
        "evecs": [asarray(ev) for ev in res.evecs],
        "kmags_lin": asarray(res.kmags_lin),
        "evecs_lin": [asarray(ev) for ev in res.evecs_lin],
        "dn_max": asarray(res.dn_max),
    }
