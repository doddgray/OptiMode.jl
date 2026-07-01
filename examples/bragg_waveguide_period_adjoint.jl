# Differentiable mode solving for a 3D waveguide that is *periodic along the
# propagation axis* (ẑ) — a Bragg / photonic-crystal-defect waveguide — including
# the gradient of the modal propagation constant with respect to the **absolute
# spatial period** Λ.
#
# This extends the 2D-cross-section adjoint of Gray, West & Ram, Opt. Express 32,
# 30541 (2024) to a fully 3D Bloch eigenproblem. In a `Grid{3}` the period is the
# z-extent of the unit cell, Λ ≡ grid.Δz, and the smoothed inverse-permittivity
# field ε⁻¹ describes one period. The Helmholtz operator depends on Λ only through
# the z-components of the reciprocal-lattice vectors g_z = m/Λ, and
# `solve_k_periodic` carries a reverse-mode rule giving ∂(kz)/∂Λ (and ∂/∂ω, ∂/∂ε⁻¹)
# at ≈one extra eigensolve.
#
# Run with a Julia environment that has OptiMode (or its MaxwellEigenmodes
# component) plus Enzyme and FiniteDifferences available.

using MaxwellEigenmodes
using MaxwellEigenmodes.DielectricSmoothing            # Grid, x, y
using LinearAlgebra, StaticArrays
using Enzyme, FiniteDifferences

const solver = KrylovKitEigsolve()

"""
Inverse-permittivity field of a Bragg waveguide: a transverse Gaussian index bump
modulated sinusoidally along z. The z modulation is defined in index space so the
structure stretches with the absolute period Λ when Λ (= grid.Δz) is varied with the
ε⁻¹ array held fixed — exactly the derivative `solve_k_periodic` differentiates.
"""
function bragg_epsi(grid::Grid{3}; ε_core=4.0, ε_clad=2.0, wx=0.5, wy=0.4, δ=0.15)
    Nx, Ny, Nz = size(grid)
    xs, ys = x(grid), y(grid)
    epsi = zeros(3, 3, Nx, Ny, Nz)
    for iz in 1:Nz
        modz = 1 + δ * cospi(2 * (iz - 1) / Nz)        # one grating period per cell
        for (iy, yy) in enumerate(ys), (ix, xx) in enumerate(xs)
            ε = ε_clad + (ε_core - ε_clad) * exp(-(xx^2 / wx^2 + yy^2 / wy^2)) * modz
            for a in 1:3
                epsi[a, a, ix, iy, iz] = inv(ε)
            end
        end
    end
    return epsi
end

ω = 1 / 1.55                                  # μm⁻¹  (λ = 1.55 μm)
Λ = 0.30                                       # μm — absolute period along ẑ
grid = Grid(4.0, 3.0, Λ, 16, 12, 8)            # transverse 4×3 μm, one period in z
epsi = bragg_epsi(grid)

# Forward solve: lowest guided Bloch mode
kmags, evecs = solve_k_periodic(ω, epsi, Λ, grid, solver; nev=1)
neff = kmags[1] / ω
println("neff = $(round(neff, digits=5)),  kz·Λ = $(round(kmags[1]*Λ, digits=4)) cycles")

# Adjoint period derivative ∂kz/∂Λ (reverse mode, one extra eigensolve) ...
kz_of_Λ(L) = solve_k_periodic(ω, epsi, L, grid, solver; nev=1)[1][1]
dkz_dΛ_adj = Enzyme.gradient(Enzyme.Reverse, kz_of_Λ, Λ)[1]
# ... checked against a finite-difference reference
dkz_dΛ_FD = central_fdm(5, 1)(kz_of_Λ, Λ)
println("∂kz/∂Λ:  adjoint = $(dkz_dΛ_adj)")
println("         finite-diff = $(dkz_dΛ_FD)")
println("         rel. error = $(abs(dkz_dΛ_adj - dkz_dΛ_FD) / abs(dkz_dΛ_FD))")

# The same call also returns ∂/∂ω and ∂/∂ε⁻¹ for free, so a dispersion- or
# bandstructure-shaping objective over (period, frequency, geometry) is fully
# differentiable for gradient-based inverse design of periodic waveguides.
