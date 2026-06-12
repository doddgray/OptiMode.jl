"""Electromagnetic eigenmode solving (wraps MaxwellEigenmodes.jl).

``solve_k`` finds guided modes at fixed frequency by Newton inversion of the
dispersion relation; ``solve_omega2`` solves the eigenvalue problem at fixed
wavevector. All frequencies/wavevectors are spatial frequencies in μm⁻¹
(``ω = 1/λ``, ``k = n_eff/λ``; units with c = 1).
"""

from __future__ import annotations

import numpy as np

from ._julia import asarray, julia, to_julia_f64
from .grids import Grid

__all__ = [
    "KrylovKitEigsolve", "GPUSolver", "MPBSolver",
    "solve_k", "solve_omega2", "k_guess",
]


def KrylovKitEigsolve():
    """Default CPU eigensolver backend (KrylovKit Lanczos/Krylov–Schur)."""
    return julia().KrylovKitEigsolve()


def GPUSolver(precision: str = "f64", device: str = "cpu"):
    """Device- and precision-generic backend (``f32``/``f64``; ``cpu``/``cuda``).

    The ``cuda`` device requires a functional CUDA GPU and ``using CUDA`` on the
    Julia side (activated automatically when available).
    """
    jl = julia()
    T = {"f32": jl.Float32, "f64": jl.Float64}[precision.lower()]
    if device == "cuda":
        jl.seval("import CUDA")
    return jl.GPUSolver(T, device=jl.Symbol(device))


def MPBSolver():
    """MPB backend: eigensolves run in Python ``meep.mpb`` via PythonCall."""
    jl = julia()
    return jl.MPBSolver()


def _eps_inv_julia(jl, eps_inv):
    return to_julia_f64(jl, np.asarray(eps_inv, dtype=np.float64))


def solve_k(omega: float, eps_inv, grid: Grid, solver=None, nev: int = 1, **kwargs):
    """Solve ``M̂(k) H = ω² H`` for the first ``nev`` guided modes at frequency ``omega``.

    - ``eps_inv``: inverse-permittivity field, shape ``(3, 3, Nx, Ny)`` (from
      :func:`optimode.inv_eps_slices`).
    - ``solver``: backend object (default :func:`KrylovKitEigsolve`).
    - keyword arguments are forwarded (``k_tol``, ``eig_tol``, ``max_eigsolves`` …).

    Returns ``(kmags, evecs)``: a length-``nev`` NumPy vector of propagation
    constants (``n_eff = kmags/omega``) and a list of complex eigenvectors
    (length ``2·Nx·Ny`` each).
    """
    jl = julia()
    if solver is None:
        solver = KrylovKitEigsolve()
    kmags, evecs = jl.solve_k(
        float(omega), _eps_inv_julia(jl, eps_inv), grid._jl, solver, nev=int(nev), **kwargs
    )
    return asarray(kmags), [asarray(ev) for ev in evecs]


def solve_omega2(k: float, eps_inv, grid: Grid, solver=None, nev: int = 1, **kwargs):
    """Solve the eigenproblem at fixed wavevector magnitude ``k`` (along ẑ).

    Returns ``(omega2, evecs)``: eigenvalues ``ω²`` and eigenvectors.
    """
    jl = julia()
    if solver is None:
        solver = KrylovKitEigsolve()
    evals, evecs = jl._om_solve_omega2(
        float(k), _eps_inv_julia(jl, eps_inv), grid._jl, solver, nev=int(nev), **kwargs
    )
    return asarray(evals), [asarray(ev) for ev in evecs]


def k_guess(omega: float, eps_inv) -> float:
    """Initial guess ``k₀ = ω·n_max`` for the Newton iteration of :func:`solve_k`."""
    jl = julia()
    return float(jl.k_guess(float(omega), _eps_inv_julia(jl, eps_inv)))
