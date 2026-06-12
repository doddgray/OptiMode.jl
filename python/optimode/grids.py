"""Finite-difference spatial grids (wraps DielectricSmoothing.Grid)."""

from __future__ import annotations

import numpy as np

from ._julia import asarray, julia

__all__ = ["Grid"]


class Grid:
    """Uniform periodic spatial grid, origin-centered.

    ``Grid(Dx, Dy, Nx, Ny)`` (2D) or ``Grid(Dx, Dy, Dz, Nx, Ny, Nz)`` (3D):
    a cell of physical size ``Dx × Dy (× Dz)`` μm discretized into
    ``Nx × Ny (× Nz)`` pixels. Mirrors ``DielectricSmoothing.Grid``.
    """

    def __init__(self, *args, _jl_obj=None):
        jl = julia()
        if _jl_obj is not None:
            self._jl = _jl_obj
        elif len(args) == 4:
            Dx, Dy, Nx, Ny = args
            self._jl = jl.Grid(float(Dx), float(Dy), int(Nx), int(Ny))
        elif len(args) == 6:
            Dx, Dy, Dz, Nx, Ny, Nz = args
            self._jl = jl.Grid(float(Dx), float(Dy), float(Dz), int(Nx), int(Ny), int(Nz))
        else:
            raise TypeError("Grid takes (Dx, Dy, Nx, Ny) or (Dx, Dy, Dz, Nx, Ny, Nz)")

    # --- shape -------------------------------------------------------------
    @property
    def shape(self) -> tuple:
        jl = julia()
        return tuple(int(n) for n in jl.size(self._jl))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    # --- coordinates and measures -------------------------------------------
    @property
    def x(self) -> np.ndarray:
        """Pixel-center x coordinates (μm)."""
        return asarray(julia().x(self._jl))

    @property
    def y(self) -> np.ndarray:
        """Pixel-center y coordinates (μm)."""
        return asarray(julia().y(self._jl))

    @property
    def z(self) -> np.ndarray:
        """Pixel-center z coordinates (μm; 3D grids)."""
        return asarray(julia().z(self._jl))

    @property
    def dx(self) -> float:
        return float(julia().δx(self._jl))

    @property
    def dy(self) -> float:
        return float(julia().δy(self._jl))

    @property
    def dV(self) -> float:
        """Pixel area (2D, μm²) / voxel volume (3D, μm³)."""
        return float(julia().δV(self._jl))

    def __repr__(self) -> str:
        return f"Grid(shape={self.shape})"
