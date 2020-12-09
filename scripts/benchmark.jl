using OptiMode, BenchmarkTools
include("mpb_example.jl") # for now, just to load ε⁻¹

H,kz = solve_k(ω,ε⁻¹,Δx,Δy,Δz)
@benchmark solve_k($ω,$ε⁻¹,$Δx,$Δy,$Δz)
# BenchmarkTools.Trial:
#   memory estimate:  250.60 MiB
#   allocs estimate:  147758
#   --------------
#   minimum time:     1.178 s (0.97% GC)
#   median time:      1.385 s (0.80% GC)
#   mean time:        1.419 s (0.79% GC)
#   maximum time:     1.728 s (0.82% GC)
#   --------------
#   samples:          4
#   evals/sample:     1

ωs = collect(0.5:0.01:0.6)
Hs,ks = solve_k(ωs,ε⁻¹,Δx,Δy,Δz)
@benchmark solve_k($ωs,$ε⁻¹,$Δx,$Δy,$Δz)
# BenchmarkTools.Trial:
#   memory estimate:  1.23 GiB
#   allocs estimate:  1120050
#   --------------
#   minimum time:     5.371 s (1.19% GC)
#   median time:      5.371 s (1.19% GC)
#   mean time:        5.371 s (1.19% GC)
#   maximum time:     5.371 s (1.19% GC)
#   --------------
#   samples:          1
#   evals/sample:     1


H,ω = solve_ω(kz,ε⁻¹,Δx,Δy,Δz)
@benchmark solve_ω($kz,$ε⁻¹,$Δx,$Δy,$Δz)
# BenchmarkTools.Trial:
#   memory estimate:  114.77 MiB
#   allocs estimate:  58115
#   --------------
#   minimum time:     541.258 ms (1.01% GC)
#   median time:      748.920 ms (0.73% GC)
#   mean time:        787.952 ms (0.78% GC)
#   maximum time:     1.127 s (0.91% GC)
#   --------------
#   samples:          7
#   evals/sample:     1

ks = collect(1.5:0.01:1.6)
Hs,ωs = solve_ω(ks,ε⁻¹,Δx,Δy,Δz)
@benchmark solve_ω($ks,$ε⁻¹,$Δx,$Δy,$Δz)
# BenchmarkTools.Trial:
#   memory estimate:  632.98 MiB
#   allocs estimate:  309718
#   --------------
#   minimum time:     2.887 s (0.89% GC)
#   median time:      2.931 s (0.80% GC)
#   mean time:        2.931 s (0.80% GC)
#   maximum time:     2.976 s (0.71% GC)
#   --------------
#   samples:          2
#   evals/sample:     1
