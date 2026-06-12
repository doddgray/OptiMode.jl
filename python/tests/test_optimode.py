"""End-to-end tests of the optimode Python wrapper against known physics."""

import numpy as np
import pytest

import optimode as om

LAM = 1.55
OMEGA = 1 / LAM


@pytest.fixture(scope="module")
def problem():
    """Small Si3N4-core rectangular waveguide in SiO2."""
    grid = om.Grid(4.0, 3.0, 32, 24)
    f_eps = om.f_eps_mats([om.Si3N4, om.SiO2])
    mat_vals = f_eps([OMEGA])
    core = om.box([0.0, 0.0], [1.6, 0.8], 1)
    sm = om.smooth_eps([core], mat_vals, (1, 2), grid)
    eps_inv, deps, ddeps = om.inv_eps_slices(sm)
    eps = np.ascontiguousarray(sm[:, :, 0])
    n2 = om.smooth_scalar([core], [om.kerr_n2(om.Si3N4, LAM), om.kerr_n2(om.SiO2, LAM)],
                          (1, 2), grid)
    return dict(grid=grid, eps=eps, eps_inv=eps_inv, deps=deps, ddeps=ddeps, n2=n2)


def test_material_values():
    assert om.index(om.SiO2, LAM) == pytest.approx(1.444, abs=2e-3)
    assert om.index(om.Si3N4, LAM) == pytest.approx(1.996, abs=2e-2)
    e_ln = om.eps(om.LiNbO3, LAM)
    assert e_ln[0, 0] == pytest.approx(e_ln[1, 1])
    assert e_ln[0, 0] != pytest.approx(e_ln[2, 2])
    # Kerr coefficients: library values; unspecified -> 0
    assert om.kerr_n2(om.Si3N4) == 2.4e-7
    assert om.kerr_n2(om.SiO2) == 2.6e-8
    assert om.kerr_n2(om.LiNbO3) == 0.0
    assert om.kerr_n2(om.with_kerr_n2(om.SiO2, 5e-8)) == 5e-8


def test_grid():
    g = om.Grid(6.0, 4.0, 64, 32)
    assert g.shape == (64, 32)
    assert g.dx == pytest.approx(6.0 / 64)
    assert g.dV == pytest.approx((6.0 / 64) * (4.0 / 32))
    assert g.x[0] == pytest.approx(-3.0)
    assert len(g.x) == 64


def test_smoothing(problem):
    eps = problem["eps"]
    assert eps.shape == (3, 3, 32, 24)
    diag = eps[0, 0]
    # bounded by the material indices; interface pixels strictly between
    assert diag.min() >= 1.444**2 - 1e-6
    assert diag.max() <= 1.996**2 + 1e-2
    assert np.any((diag > 1.444**2 + 0.01) & (diag < 1.996**2 - 0.01))
    # Kerr map mirrors the geometry
    n2 = problem["n2"]
    assert n2.max() == pytest.approx(2.4e-7, rel=1e-9)
    assert n2.min() == pytest.approx(2.6e-8, rel=1e-9)


def test_solve_and_analysis(problem):
    g, eps_inv, deps = problem["grid"], problem["eps_inv"], problem["deps"]
    kmags, evecs = om.solve_k(OMEGA, eps_inv, g, nev=1, k_tol=1e-10)
    neff = kmags[0] / OMEGA
    assert 1.444 < neff < 1.996          # guided mode
    assert evecs[0].shape == (2 * 32 * 24,)

    ng = om.group_index(kmags[0], evecs[0], OMEGA, eps_inv, deps, g)
    assert neff < ng < 2.5               # SiN guide: ng ≈ 2.1

    ng2, gvd = om.ng_gvd(OMEGA, kmags[0], evecs[0], eps_inv, deps, problem["ddeps"], g)
    assert ng2 == pytest.approx(ng, rel=1e-6)
    assert np.isfinite(gvd)

    E = om.E_field(kmags[0], evecs[0], eps_inv, deps, g)
    assert E.shape == (3, 32, 24)
    pol = om.rel_power_xyz(problem["eps"], E)
    assert pol.argmax() == 0             # fundamental quasi-TE (x-polarized)
    assert om.count_E_nodes(E / np.abs(E[:2]).max(), problem["eps"], 1) == (0, 0)

    # round trip: omega2 at k(omega) returns omega^2
    w2, _ = om.solve_omega2(kmags[0], eps_inv, g, nev=1)
    assert w2[0] == pytest.approx(OMEGA**2, rel=1e-6)


def test_kerr(problem):
    g, eps_inv, deps, n2 = (problem[k] for k in ("grid", "eps_inv", "deps", "n2"))
    res = om.solve_k_kerr(OMEGA, 5.0, eps_inv, deps, n2, g, nev=1,
                          k_tol=1e-12, eig_tol=1e-12)
    dneff1 = (res["kmags"][0] - res["kmags_lin"][0]) / OMEGA
    assert dneff1 > 0
    res2 = om.solve_k_kerr(OMEGA, 10.0, eps_inv, deps, n2, g, nev=1,
                           k_tol=1e-12, eig_tol=1e-12)
    dneff2 = (res2["kmags"][0] - res2["kmags_lin"][0]) / OMEGA
    assert dneff2 == pytest.approx(2 * dneff1, rel=5e-2)   # linear in P

    # intensity normalization and SPM cross-check
    I = om.mode_intensity(res["kmags_lin"][0], res["evecs_lin"][0], eps_inv, g, 5.0)
    assert I.sum() * g.dV == pytest.approx(5.0, rel=1e-9)
    Aeff = 5.0**2 / ((I**2).sum() * g.dV)
    assert 0.2 < dneff1 / (2.4e-7 * 5.0 / Aeff) < 1.5


def test_sweeps(tmp_path):
    setup = tmp_path / "setup.jl"
    setup.write_text("""
function make_problem(p)
    grid = Grid(4.0, 3.0, 16, 12)
    Nx, Ny = size(grid)
    eps = zeros(3,3,Nx,Ny); deps = zeros(3,3,Nx,Ny)
    for (iy,yy) in enumerate(y(grid)), (ix,xx) in enumerate(x(grid))
        n = (abs(xx) <= 0.8 && abs(yy) <= 0.4) ? 1.996 : 1.444
        for a in 1:3
            eps[a,a,ix,iy] = n^2
            deps[a,a,ix,iy] = 0.3
        end
    end
    return (; ε⁻¹=sliceinv_3x3(eps), ∂ε_∂ω=deps, ∂²ε_∂ω²=zeros(3,3,Nx,Ny), grid)
end
""")
    params = om.param_grid(ω=[1 / 1.6, 1 / 1.5], P=0.0)
    assert len(params) == 2 and params[0]["ω"] == pytest.approx(1 / 1.6)

    batch = om.deploy_batch(str(setup), params, name="pytest", dir=str(tmp_path / "b"),
                            nev=1, backend="local", solver_kwargs={"k_tol": 1e-9})
    import time
    t0 = time.time()
    st = batch.status(verbose=False)
    while st["pending"] > 0 and time.time() - t0 < 600:
        time.sleep(5)
        st = batch.status(verbose=False)
    assert st["done"] == 2 and st["failed"] == 0

    rows = batch.gather(save=True)
    assert len(rows) == 2
    assert all(1.444 < r["neff"] < 1.996 for r in rows)
    # persistence round trips
    re = om.load_summary(str(tmp_path / "b" / "summary.csv"))
    assert len(re) == 2
    b2 = om.load_batch(str(tmp_path / "b"))
    assert len(b2) == 2
