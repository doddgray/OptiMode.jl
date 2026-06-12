"""Geometry definition and sub-pixel dielectric smoothing (wraps DielectricSmoothing.jl).

Shapes carry a *material index* selecting a column of the material data passed to
:func:`smooth_eps` / :func:`smooth_scalar`; points outside all shapes belong to the
background material (index ``len(shapes) + 1`` by the ``minds`` convention).
"""

from __future__ import annotations

import numpy as np

from ._julia import asarray, julia, to_julia_f64
from .grids import Grid

__all__ = [
    "box", "polygon", "ball", "material_shape",
    "smooth_eps", "smooth_scalar", "inv_eps_slices",
]


def material_shape(shape, mat_idx: int):
    """Wrap a GeometryPrimitives shape with a material index (``MaterialShape``)."""
    return julia().MaterialShape(shape, int(mat_idx))


def box(center, size, mat_idx: int, axes=None):
    """Axis-aligned (or rotated) rectangular ``Cuboid`` carrying material index ``mat_idx``.

    ``center`` and ``size`` are length-2 (2D) or length-3 (3D) sequences in μm;
    ``axes`` is an optional principal-axes matrix (defaults to identity).
    """
    jl = julia()
    c = np.asarray(center, dtype=np.float64)
    s = np.asarray(size, dtype=np.float64)
    A = np.eye(len(c)) if axes is None else np.asarray(axes, dtype=np.float64)
    shp = jl.Cuboid(to_julia_f64(jl, c), to_julia_f64(jl, s), to_julia_f64(jl, A))
    return material_shape(shp, mat_idx)


def polygon(vertices, mat_idx: int):
    """2D polygon from a ``(2, K)`` (or ``(K, 2)``) vertex array, counter-clockwise."""
    jl = julia()
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape[0] != 2 and v.shape[1] == 2:
        v = v.T
    shp = jl.Polygon(to_julia_f64(jl, v))
    return material_shape(shp, mat_idx)


def ball(center, radius: float, mat_idx: int):
    """Circle (2D) / sphere (3D) of given ``center`` and ``radius`` (μm)."""
    jl = julia()
    shp = jl.Ball(to_julia_f64(jl, np.asarray(center, dtype=np.float64)), float(radius))
    return material_shape(shp, mat_idx)


def _shapes_tuple(jl, shapes):
    return jl.seval("(xs...,) -> tuple(xs...)")(*list(shapes))


def _minds_tuple(jl, minds):
    return jl.seval("(xs...,) -> tuple(map(Int, xs)...)")(*[int(m) for m in minds])


def smooth_eps(shapes, mat_vals, minds, grid: Grid) -> np.ndarray:
    """Sub-pixel (Kottke) smoothing of dielectric tensors onto ``grid``.

    - ``shapes``: foreground-first sequence of shapes from :func:`box` /
      :func:`polygon` / :func:`ball`.
    - ``mat_vals``: per-material data columns ``vcat(vec(ε), vec(∂ωε), vec(∂²ωε))``,
      e.g. from :func:`optimode.f_eps_mats`; shape ``(27, n_materials)``.
    - ``minds``: material index for each shape and, last, the background, e.g.
      ``(1, 2)`` for one shape + background.

    Returns a ``(3, 3, 3, Nx, Ny)`` array whose third axis indexes
    ``(ε, ∂ε/∂ω, ∂²ε/∂ω²)``. Use :func:`inv_eps_slices` to split it into the
    eigensolver inputs.
    """
    jl = julia()
    sm = jl.smooth_ε(
        _shapes_tuple(jl, shapes),
        to_julia_f64(jl, np.asarray(mat_vals, dtype=np.float64)),
        _minds_tuple(jl, minds),
        grid._jl,
    )
    return asarray(sm)


def smooth_scalar(shapes, vals, minds, grid: Grid) -> np.ndarray:
    """Volume-fraction smoothing of per-material scalars (e.g. Kerr n₂ maps).

    ``vals[j]`` is material ``j``'s scalar value; returns an array shaped like the
    grid.
    """
    jl = julia()
    out = jl.smooth_scalar(
        _shapes_tuple(jl, shapes),
        to_julia_f64(jl, np.asarray(vals, dtype=np.float64)),
        _minds_tuple(jl, minds),
        grid._jl,
    )
    return asarray(out)


def inv_eps_slices(sm: np.ndarray):
    """Split :func:`smooth_eps` output into ``(eps_inv, deps_dom, d2eps_dom2)``.

    Returns the inverse-permittivity field ``ε⁻¹`` (input of
    :func:`optimode.solve_k`) and the first/second frequency-derivative fields.
    """
    jl = julia()
    eps = np.ascontiguousarray(sm[:, :, 0])
    deps = np.ascontiguousarray(sm[:, :, 1])
    ddeps = np.ascontiguousarray(sm[:, :, 2])
    eps_inv = asarray(jl.sliceinv_3x3(to_julia_f64(jl, eps)))
    return eps_inv, deps, ddeps
