using LinearAlgebra, StaticArrays, PyCall, FiniteDifferences, Plots  #FFTW, BenchmarkTools, LinearMaps, IterativeSolvers, Roots, GeometryPrimitives
SHM3 = SHermitianCompact{3,Float64,6}
mp = pyimport("meep")
mpb = pyimport("meep.mpb")

"""
################################################################################
#																			   #
#		         Compare MPB and Julia solutions and gradients of 			   #
#		                Ridge Waveguide Dispersion Model 	           		   #
#																			   #
################################################################################
"""

# Default parameter values
p_def = [
        1.55,               #   wavelength              `λ`             [μm]
        1.7,                #   top ridge width         `w_top`         [μm]
        0.7,                #   ridge thickness         `t_core`        [μm]
        π/14.,              #   ridge sidewall angle    `θ`             [radian]
        2.4,                #   core index              `n_core`        [1]
        1.4,                #   substrate index         `n_subs`        [1]
        ]

function nng_rwg_mpb(p=p_def;Δx=6.,Δy=4.,Δz=1.,Nx=128,Ny=128,Nz=1,edge_gap=0.5,band_idx=1,tol=1e-5)
    λ,w,t_core,θ,n_core,n_subs = p
    ω           =   1 / λ
    nk          = 10
    n_bands     = 1
    res         = mp.Vector3( ( Nx / Δx ), ( Ny / Δy ), 1) #mp.Vector3(Int(Nx/Δx),Int(Ny/Δy),1) #16
    n_guess = 0.9 * n_core
    n_min = n_subs
    n_max = n_core
    t_subs = (Δy -t_core - edge_gap )/2.
    c_subs_y = -Δy/2. + edge_gap/2. + t_subs/2.
    # Set up MPB modesolver, use find-k to solve for one eigenmode `H` with prop. const. `k` at specified temporal freq. ω
    k_pts = mp.interpolate(nk, [mp.Vector3(0.05, 0, 0), mp.Vector3(0.05*nk, 0, 0)] )
    lat = mp.Lattice(size=mp.Vector3(Δx, Δy,0))
    tanθ = tan(θ)
    tcore_tanθ = t_core*tanθ
    w_bottom = w + 2*tcore_tanθ
    verts = [ mp.Vector3( w/2., t_core/2., -5.), mp.Vector3(-w/2., t_core/2., -5.),mp.Vector3(-w_bottom/2., -t_core/2., -5.), mp.Vector3(w_bottom/2., -t_core/2., -5.)  ]
    # verts = [ mp.Vector3(-w/2., -t_core/2., -5.),mp.Vector3(w, 2*t_core, -5.), mp.Vector3(w, -t_core/2., -5.)  ]
    core = mp.Prism(verts,
                       10.0,
                       axis=mp.Vector3(0.,0.,1.),
                       material=mp.Medium(index=n_core),
                      )
    subs = mp.Block(size=mp.Vector3(Δx-edge_gap, t_subs , 10.0),
                    center=mp.Vector3(0, c_subs_y, 0),
                    material=mp.Medium(index=n_subs),
                    )

    ms = mpb.ModeSolver(geometry_lattice=lat,
                        geometry=[core,subs],
                        k_points=k_pts,
                        resolution=res,
                        num_bands=n_bands,
                        default_material=mp.vacuum)

    ms.init_params(mp.NO_PARITY, false)

    kz = ms.find_k(mp.NO_PARITY,             # parity (meep parity object)
                      ω,                    # ω at which to solve for k
                      band_idx,                        # band_min (find k(ω) for bands
                      band_idx,                        # band_max  band_min:band_max)
                      mp.Vector3(0, 0, 1),      # k direction to search
                      tol,                     # fractional k error tolerance
                      n_guess*ω,              # kmag_guess, |k| estimate
                      n_min*ω,                # kmag_min (find k in range
                      n_max*ω,               # kmag_max  kmag_min:kmag_max)
    )[1]
    neff = kz / ω
    ng = 1 / ms.compute_one_group_velocity_component(mp.Vector3(0, 0, 1), band_idx)
    # ng = 1 / ms.compute_group_velocity_component(mp.Vector3(0, 0, 1))
    return [neff, ng]
end

∇nng_rwg_mpb_FD(p;Δx=6.,Δy=4.,Δz=1.,Nx=128,Ny=128,Nz=1,edge_gap=0.5,band_idx=1,tol=1e-5) = FiniteDifferences.jacobian(central_fdm(2,1),x->nng_rwg_mpb(x;Δx,Δy,Δz,Nx,Ny,Nz,edge_gap,band_idx,tol),p)
nng_rwg_mpb(p_def)
∇nng_rwg_mpb_FD(p_def)
wt_rwg_mpb(wt;NN=128,bi=1,tol=1e-5) = nng_rwg_mpb([p_def[1], wt[1], wt[2], p_def[4:6]...];Nx=NN,Ny=NN,band_idx=bi,tol)
∇wt_rwg_mpb_FD(wt;NN=128,bi=1,tol=1e-5) = FiniteDifferences.jacobian(central_fdm(2,1),x->wt_rwg_mpb(x;NN,bi,tol),wt)[1]
wt_rwg_mpb([1.0,0.7];NN=64,bi=1,tol=1e-5)
grad_nng = ∇wt_rwg_mpb_FD([1.0,0.7];NN=64,bi=1,tol=1e-5)

function wt_rwg_mpb_sweep(ws,ts,Ns;bi=1,tol=1e-5)
    nw = length(ws); nt = length(ts); nN = length(Ns);
    nng_mpb = zeros(Float64,(2,nw,nt,nN))
    ∇nng_mpb = zeros(Float64,(2,2,nw,nt,nN))
    for wind=1:nw, tind=1:nt, Nind=1:nN
        nng_mpb[:,wind,tind,Nind] = wt_rwg_mpb([ws[wind],ts[tind]];NN=Ns[Nind],bi,tol)
        @views ∇nng_mpb[:,:,wind,tind,Nind] .= ∇wt_rwg_mpb_FD([ws[wind],ts[tind]];NN=Ns[Nind],bi,tol)
    end
    return nng_mpb, ∇nng_mpb
end

# function wt_rwg_sweep(ws,ts,Ns;bi=1,tol=1e-5,name="rwg_ng_bm_N",path="/home/dodd/github/OptiMode/test/")
#     bm = [ @benchmark(outs[$i] = ngom_rwg($ks,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$Ns[$i],$Ns[$i],1))  for i=1:length(Ns) ]
#     ns = [o[1] for o in outs]
#     ngs = [o[2] for o in outs]
#     # bm = [ @benchmark(ngom_rwg(1.6,$w,$t_core,$edge_gap,$n_core,$n_subs,$Δx,$Δy,$Δz,$nn,$nn,1))  for nn=Ns ]
#     df = DataFrame( N = Ns,
#                     Nsq = (Ns.^2),
#                     n = ns,
#                     ng = ngs,
#                     time = (time.(bm).*1e-9),
#                     gctime_frac = (gctime.(bm) ./ time.(bm)),
#                     mem = memory.(bm),
#                     allocs = allocs.(bm),
#                    )
#     CSV.write(*(path,name,".csv"), df)
#     # plot_bm_single(df)
#     df
# end

########################################
##### w-t-N sweep 1 (MPB only) #########
########################################
# ws = collect(0.8:0.1:1.7)
# ts = collect(0.5:0.1:1.3)
# Ns = Int.(2 .^(collect(5:8)))
# nng_mpb, ∇nng_mpb = wt_rwg_mpb_sweep(ws,ts,Ns;bi=1)
# ws1 = collect(0.8:0.1:1.7)
# ts1 = collect(0.5:0.1:1.3)
# Ns1 = Int.(2 .^(collect(5:8)))
# nng_mpb1 = copy(nng_mpb)
# ∇nng_mpb1 = copy(∇nng_mpb)
# n_mpb1 = nng_mpb1[1,:,:,:]
# ng_mpb1 = nng_mpb1[2,:,:,:]
# ∂n∂w_mpb1 = ∇nng_mpb1[1,1,:,:,:]
# ∂n∂t_mpb1 = ∇nng_mpb1[1,2,:,:,:]
# ∂ng∂w_mpb1 = ∇nng_mpb1[2,1,:,:,:]
# ∂ng∂t_mpb1 = ∇nng_mpb1[2,2,:,:,:]
# mpb_fname1="rwg_mpb_wt1.h5"
# # path="/home/dodd/github/OptiMode/test/"
# h5open(mpb_fname1, "cw") do file
#     write(file, "ws", ws1)  # alternatively, say "@write file A"
#     write(file, "ts", ts1)
#     write(file, "Ns", Ns1)
#     write(file, "nng", nng_mpb1)
#     write(file, "∇nng", ∇nng_mpb1)
#     write(file, "∂n∂w", ∂n∂w_mpb1)
#     write(file, "∂n∂t", ∂n∂t_mpb1)
#     write(file, "∂ng∂w", ∂ng∂w_mpb1)
#     write(file, "∂ng∂t", ∂ng∂t_mpb1)
# end
# ∇nng_mpb1_read = h5open(mpb_fname1, "r") do file
#     read(file, "∇nng")
# end



function plot_wt_sweep(ws,ts,Ns,nng,∇nng;tind_w=4,size=(800,800))
    n = nng[1,:,:,:]
    ng = nng[2,:,:,:]
    ∂n∂w = ∇nng[1,1,:,:,:]
    ∂n∂t = ∇nng[1,2,:,:,:]
    ∂ng∂w = ∇nng[2,1,:,:,:]
    ∂ng∂t = ∇nng[2,2,:,:,:]
    # w slices
    p_n_w = plot(ws,
                n[:,tind_w,1],
                label="N=$(Ns[1])",
                xlabel= "top width [μm]",
                ylabel= "effective index n",
                legend= :bottomright,
                )
    p_ng_w = plot(ws,
                ng[:,tind_w,1],
                label="N=$(Ns[1])",
                xlabel= "top width [μm]",
                ylabel= "eff. group index ng",
                legend= false,
                )
    p_∂n∂w_w = plot(ws,
                ∂n∂w[:,tind_w,1],
                label="N=$(Ns[1])",
                xlabel= "top width [μm]",
                ylabel= "∂n/∂w [μm⁻¹]",
                legend= false,
                )
    p_∂ng∂w_w = plot(ws,
                ∂ng∂w[:,tind_w,1],
                label="N=$(Ns[1])",
                xlabel= "top width [μm]",
                ylabel= "∂ng/∂w [μm⁻¹]",
                legend= false,
                )
    p_∂n∂t_w = plot(ws,
                ∂n∂t[:,tind_w,1],
                label="N=$(Ns[1])",
                xlabel= "top width [μm]",
                ylabel= "∂n/∂t [μm⁻¹]",
                legend= false,
                )
    p_∂ng∂t_w = plot(ws,
                ∂ng∂t[:,tind_w,1],
                label="N=$(Ns[1])",
                xlabel= "top width [μm]",
                ylabel= "∂ng/∂t [μm⁻¹]",
                legend= false,
                )
    for Nind=2:length(Ns)
        plot!(      p_n_w,
                    ws,
                    n[:,tind_w,Nind],
                    label="N=$(Ns[Nind])",
                    xlabel= "top width [μm]",
                    ylabel= "effective index n",
                    )
        plot!(      p_ng_w,
                    ws,
                    ng[:,tind_w,Nind],
                    label="N=$(Ns[Nind])",
                    xlabel= "top width [μm]",
                    ylabel= "eff. group index ng",
                    )
        plot!(      p_∂n∂w_w,
                    ws,
                    ∂n∂w[:,tind_w,Nind],
                    label="N=$(Ns[Nind])",
                    xlabel= "top width [μm]",
                    ylabel= "∂n/∂w [μm⁻¹]",
                    )
        plot!(      p_∂ng∂w_w,
                    ws,
                    ∂ng∂w[:,tind_w,Nind],
                    label="N=$(Ns[Nind])",
                    xlabel= "top width [μm]",
                    ylabel= "∂ng/∂w [μm⁻¹]",
                    )
        plot!(      p_∂n∂t_w,
                    ws,
                    ∂n∂t[:,tind_w,Nind],
                    label="N=$(Ns[Nind])",
                    xlabel= "top width [μm]",
                    ylabel= "∂n/∂t [μm⁻¹]",
                    )
        plot!(      p_∂ng∂t_w,
                    ws,
                    ∂ng∂t[:,tind_w,Nind],
                    label="N=$(Ns[Nind])",
                    xlabel= "top width [μm]",
                    ylabel= "∂ng/∂t [μm⁻¹]",
                    )
    end
    l_w = @layout   [   a   b
                        c   d
                        e   f   ]
    plot(   p_n_w,      p_ng_w,
            p_∂n∂w_w,   p_∂ng∂w_w,
            p_∂n∂t_w,   p_∂ng∂t_w,
            layout = l_w,
            size=size,
            )
end

plot_wt_sweep(ws,ts,Ns,nng_mpb,∇nng_mpb;tind_w=2)
