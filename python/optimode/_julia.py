"""Julia runtime bootstrap for the optimode Python package.

The first call to :func:`julia` starts an embedded Julia process via JuliaCall,
activates the OptiMode.jl project and loads the ``OptiMode`` umbrella package
(which re-exports MaterialDispersion, DielectricSmoothing, MaxwellEigenmodes,
ModeAnalysis and ModeSweeps). Everything is cached, so subsequent calls are free.

Configuration (environment variables, all optional):

- ``OPTIMODE_JULIA_PROJECT``: path of the Julia project to activate. Defaults to
  the OptiMode.jl repository root when this package is used from a checkout
  (``python/`` inside the repo), otherwise to a project where ``OptiMode`` is
  installed.
- ``PYTHON_JULIAPKG_EXE``: path to the ``julia`` executable. If unset, ``julia``
  is searched on ``PATH`` and in ``~/.juliaup/bin``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

_jl = None


def _default_project() -> str | None:
    env = os.environ.get("OPTIMODE_JULIA_PROJECT")
    if env:
        return env
    # python/optimode/_julia.py → repository root (when used from a checkout)
    root = Path(__file__).resolve().parents[2]
    if (root / "Project.toml").exists() and (root / "lib").exists():
        return str(root)
    return None


def _find_julia() -> str | None:
    exe = os.environ.get("PYTHON_JULIAPKG_EXE") or shutil.which("julia")
    if exe is None:
        cand = Path.home() / ".juliaup" / "bin" / "julia"
        if cand.exists():
            exe = str(cand)
    if exe is None:
        return None
    # juliaup installs a launcher shim; juliacall needs the real binary (it locates
    # the sysimage relative to the executable). Ask Julia for its true BINDIR once.
    if ".juliaup" in str(exe) or Path(exe).is_symlink():
        import subprocess

        try:
            bindir = subprocess.run(
                [exe, "--startup-file=no", "-e", "print(Sys.BINDIR)"],
                capture_output=True, text=True, timeout=120, check=True,
            ).stdout.strip()
            real = Path(bindir) / "julia"
            if real.exists():
                exe = str(real)
        except Exception:
            pass
    return exe


def _preload_julia_openssl(exe: str, project: str | None) -> None:
    """Best-effort fix for the libssl/libcrypto version clash between Julia's OpenSSL
    artifact and the system OpenSSL already linked into the Python process: load the
    artifact libraries globally *before* Julia (or Python's ``ssl``) loads anything.
    Harmless no-op when it fails; set ``LD_PRELOAD`` to the artifact libs as a
    fallback if OpenSSL ``InitError``s persist.
    """
    import ctypes
    import subprocess

    proj = f"--project={project}" if project else "--startup-file=no"
    code = (
        'id = Base.PkgId(Base.UUID("458c3c95-2e84-50aa-8efc-19380b2a3a95"), "OpenSSL_jll");'
        "m = Base.require(id); print(m.libcrypto_path, ';', m.libssl_path)"
    )
    try:
        out = subprocess.run(
            [exe, "--startup-file=no", proj, "-e", code],
            capture_output=True, text=True, timeout=300, check=True,
        ).stdout.strip()
        for p in out.split(";"):
            if p and Path(p).exists():
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


def julia():
    """Return the (lazily initialized) JuliaCall ``Main`` module with OptiMode loaded."""
    global _jl
    if _jl is not None:
        return _jl

    exe = _find_julia()
    if exe and "PYTHON_JULIAPKG_EXE" not in os.environ:
        os.environ["PYTHON_JULIAPKG_EXE"] = str(exe)
    if exe:
        _preload_julia_openssl(exe, _default_project())

    from juliacall import Main as jl  # starts Julia

    project = _default_project()
    if project is not None:
        # Stack the OptiMode project on the load path (rather than activating it) so
        # that JuliaCall's own environment keeps providing PythonCall — required by
        # juliacall itself and by the MPB backend extension.
        jl.seval(f'pushfirst!(LOAD_PATH, raw"{project}")')
    jl.seval("using OptiMode")
    jl.seval("using LinearAlgebra")
    jl.seval(
        "using OptiMode.DielectricSmoothing.GeometryPrimitives: Cuboid, Polygon, Ball"
    )
    # Helper shims (defined once in Main, prefixed to avoid collisions):
    jl.seval("_om_f64(A) = Array{Float64}(A)")
    # ASCII aliases for combining-character Julia names (bytes via \\u escapes so the
    # Python source encoding can never corrupt them):
    jl.seval('const _om_nng = getproperty(Main, Symbol("nn\\u0302g"))')        # nn̂g
    jl.seval('const _om_ngvd = getproperty(Main, Symbol("n\\u011dvd"))')       # nĝvd
    jl.seval('const _om_solve_omega2 = getproperty(Main, Symbol("solve_\\u03c9\\u00b2"))')  # solve_ω²
    jl.seval('const _om_Efield = getproperty(Main, Symbol("E\\u20d7"))')       # E⃗
    jl.seval("_om_nt(pairs_list) = (; (Symbol(string(first(p))) => last(p) for p in pairs_list)...)")
    jl.seval("""
        _om_params(ps) = [ _om_nt(p) for p in ps ]
    """)
    jl.seval("""
        function _om_rows_py(rows)
            PythonCall.pylist([
                PythonCall.pydict(Dict(String(k) => getfield(r, k) for k in keys(r)))
                for r in rows
            ])
        end
    """)
    jl.seval("import PythonCall")
    _jl = jl
    return _jl


def asarray(x, copy: bool = True) -> np.ndarray:
    """Convert a Julia array (JuliaCall wrapper) to a NumPy array."""
    a = np.asarray(x)
    return a.copy() if copy else a


def to_julia_f64(jl, x) -> "object":
    """Convert array-like input to a Julia ``Array{Float64}`` (index semantics preserved)."""
    return jl._om_f64(np.asarray(x, dtype=np.float64))


def kwargs_to_julia(jl, kwargs: dict):
    """Convert a Python dict to a Julia NamedTuple (preserving insertion order)."""
    return jl._om_nt(list(kwargs.items()))
