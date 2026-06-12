"""Material dispersion models (wraps MaterialDispersion.jl).

Library materials are exposed under ASCII names (``Si3N4``, ``SiO2``, ``LiNbO3``,
``MgO_LiNbO3``, ``LiB3O5``, ``Si``, ``Ge``, ``alpha_Al2O3``, ``Vacuum``); they are
opaque Julia ``Material`` objects that all functions here accept. All wavelengths are
vacuum wavelengths in μm, frequencies are ``ω = 1/λ`` in μm⁻¹ (units with c = 1),
Kerr coefficients ``n₂`` are in μm²/W.
"""

from __future__ import annotations

import numpy as np

from ._julia import asarray, julia

__all__ = [
    "Si3N4", "SiO2", "LiNbO3", "MgO_LiNbO3", "LiB3O5", "Si", "Ge",
    "alpha_Al2O3", "Vacuum",
    "eps", "index", "nng", "ngvd", "eps_fn",
    "kerr_n2", "with_kerr_n2", "rotate",
    "f_eps_mats", "eps_views",
]

# Julia-side names of the library materials (unicode → ASCII Python attribute)
_MATERIAL_NAMES = {
    "Si3N4": "Si₃N₄",
    "SiO2": "SiO₂",
    "LiNbO3": "LiNbO₃",
    "MgO_LiNbO3": "MgO_LiNbO₃",
    "LiB3O5": "LiB₃O₅",
    "Si": "silicon",
    "Ge": "germanium",
    "alpha_Al2O3": "αAl₂O₃",
    "Vacuum": "Vacuum",
}


def __getattr__(name: str):  # lazy material access (PEP 562)
    if name in _MATERIAL_NAMES:
        return getattr(julia(), _MATERIAL_NAMES[name])
    raise AttributeError(name)


def eps(mat, lam: float) -> np.ndarray:
    """3×3 relative-permittivity tensor ε(λ) of ``mat`` at vacuum wavelength ``lam`` (μm)."""
    jl = julia()
    return asarray(jl.ε_fn(mat)(float(lam)))


def index(mat, lam: float, axis: int = 0) -> float:
    """Refractive index ``n = sqrt(ε[axis, axis])`` at wavelength ``lam`` (μm)."""
    return float(np.sqrt(eps(mat, lam)[axis, axis]))


def nng(mat, lam: float) -> np.ndarray:
    """Group-index-weighted tensor ``∂(ωε)/∂ω`` (3×3) at wavelength ``lam`` (μm)."""
    jl = julia()
    return asarray(jl._om_nng(mat, float(lam)))


def ngvd(mat, lam: float) -> np.ndarray:
    """Second frequency derivative ``∂²(ωε)/∂ω²`` (3×3) at wavelength ``lam`` (μm)."""
    jl = julia()
    return asarray(jl._om_ngvd(mat, float(lam)))


def eps_fn(mat):
    """Return a Python callable ``lam -> ε`` (3×3 NumPy array) for ``mat``."""
    jl = julia()
    f = jl.ε_fn(mat)
    return lambda lam: asarray(f(float(lam)))


def kerr_n2(mat, lam: float = 1.55) -> float:
    """Kerr coefficient n₂ of ``mat`` in μm²/W at wavelength ``lam`` (0.0 if unspecified)."""
    jl = julia()
    return float(jl.kerr_n2(mat, float(lam)))


def with_kerr_n2(mat, n2: float):
    """Copy of ``mat`` with its Kerr coefficient model set to the constant ``n2`` (μm²/W)."""
    jl = julia()
    return jl.with_kerr_n2(mat, float(n2))


def rotate(mat, R, name: str = "rotated_material"):
    """Material with tensor models rotated by the 3×3 rotation matrix ``R``."""
    jl = julia()
    Rj = jl._om_f64(np.asarray(R, dtype=np.float64))
    return jl.rotate(mat, Rj, name=jl.Symbol(name))


def f_eps_mats(mats, params=("ω",)):
    """Generated multi-material dispersion function.

    Returns a Python callable ``p -> values`` mapping a parameter vector (frequency
    ``ω = 1/λ`` first, e.g. ``[1/1.55]`` or ``[1/1.55, T]``) to the flat
    material-major data array of ``(ε, ∂ωε, ∂²ωε)`` triples — the ``mat_vals``
    input of :func:`optimode.smooth_eps`. Split with :func:`eps_views`.
    """
    jl = julia()
    syms = jl.seval("(xs...,) -> tuple(map(Symbol, xs)...)")(*[str(p) for p in params])
    f = jl.seval("(mats, syms) -> first(_f_ε_mats(collect(mats), syms))")(list(mats), syms)
    return lambda p: asarray(f(jl._om_f64(np.asarray(p, dtype=np.float64))))


def eps_views(values: np.ndarray, n_mats: int):
    """Split flat dispersion data into per-material 3×3 arrays.

    Returns ``(eps, deps, ddeps)``: three lists (one entry per material) of 3×3
    NumPy arrays holding ε, ∂ε/∂ω and ∂²ε/∂ω².
    """
    v = np.asarray(values).reshape(-1)
    out = ([], [], [])
    for m in range(n_mats):
        base = 27 * m
        for i in range(3):
            out[i].append(v[base + 9 * i: base + 9 * (i + 1)].reshape(3, 3, order="F"))
    return out
