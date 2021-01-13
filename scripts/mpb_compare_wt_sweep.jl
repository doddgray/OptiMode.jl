include("mpb_compare.jl")

## configure swept-parameter data collection
ws = collect(0.8:0.1:1.7)
ts = collect(0.5:0.1:1.3)
Ns = Int.(ceil.(2 .^ collect(5:0.5:8)))
@show sw_name = "wt_OM"

# small sweep ranges for testing the script
# @show ws = collect(0.8:0.3:1.7)
# @show ts = collect(0.5:0.3:1.3)
# @show Ns = Int.(ceil.(2 .^ collect(6:1:7)))
# @show sw_name = "wt_OM_test"

@show pω_def
@show nw = length(ws)
@show nt = length(ts)
@show nN = length(Ns)
@show np = length(pω_def)
##

nng_OM, ∇nng_OM_AD, ∇nng_OM_FD = wtω_rwg_OM_sweep(sw_name, ws, ts, Ns;
    bi = 1,
    tol = 1e-8,
    data_dir= "/home/gridsan/dodd/data/OptiMode/mpb_compare_rwg/", # "/home/dodd/data/OptiMode/mpb_compare_rwg/",
    dt_fmt=dateformat"Y-m-d--H-M-S",
    extension=".h5",
    )

# ∇nng_err_OM = abs.(∇nng_OM_FD .- ∇nng_OM_AD) ./ abs.(∇nng_OM_FD)

# first run:
# /home/dodd/data/OptiMode/mpb_compare_rwg/wt_OM_test_2021-1-13--14-21-40.h5
