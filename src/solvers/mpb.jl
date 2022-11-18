export find_k, load_evecs, load_epsilon, save_epsilon, DEFAULT_MPB_LOGPATH, DEFAULT_MPB_EPSPATH
export MPB_Solver, n_range #, mp, mpb, np

# mutable struct MPB_Solver <: AbstractEigensolver end

mutable struct MPB_Solver{L<:AbstractLogger} <: AbstractEigensolver{L}
    logger::L
end

MPB_Solver() = MPB_Solver(NullLogger())

# mutable struct MPB_Solver <: AbstractEigensolver
#     num_bands=2,band_min=1,band_max=num_bands,eig_tol=1e-8,k_tol=1e-8,
#     k_dir=[0.,0.,1.]

# end


# using Parameters

# @with_kw mutable struct MPB_Solver{T} <: AbstractEigensolver
# 	"Number of bands (modes) to find"
#     num_bands::Int64 = 2

#     "Lowest band number to find"
#     band_min::Int64 = 1

#     "Highest band number to find"
#     band_max::Int64 = 2
    
#     "Absolute k magnitude tolerance for outer (Newton) solver"
# 	k_tol::T = 1e-9

# 	"Absolute eigenvalue tolerance for inner (eigen) solver"
# 	eig_tol::T = 1e-9

# 	"Number of restarts"
# 	k_dir::Vec{T} = [0.,0.,1]

# 	"Path to directory where solver log and field data should be saved"
# 	data_path::String = pwd()

#     "filename prefix for saving solver log and field data"
# 	filename_prefix::String = "f01"

# 	"Display information during iterations"
# 	verbose::Bool = false

# 	"Record information"
# 	log::Bool = true

# 	"Save eigenvector data"
# 	save_evecs::Bool = false

#     "Save electric field data"
# 	save_efield::Bool = false
# end

# function rrule(::typeof(solve_k), ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::AbstractEigensolver;nev=1,
# 	max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
# 	f_filter=nothing) where {ND,T<:Real} 

function solve_k(ω::T,ε⁻¹::AbstractArray{T},grid::Grid{ND,T},solver::MPB_Solver; nev=1,
    max_eigsolves=60,maxiter=100,k_tol=1e-8,eig_tol=1e-8,log=false,kguess=nothing,Hguess=nothing,
    f_filter=nothing,overwrite=false) where {ND,T<:Real}
    ε = sliceinv_3x3(ε⁻¹)
    ks,evecs = find_k(ω,ε,grid;num_bands=nev,overwrite,k_tol,eig_tol)
    return ks, evecs
end

# function solve_k(ω::,ms::ModeSolver{ND,T},solver::MPB_Solver;nev=1,eigind=1,maxiter=100,tol=1e-8,log=false,f_filter=nothing,kwargs...) where {ND,T<:Real}
# 	ks,evecs = find_k(ω,ε,grid;num_bands=nev,kwargs...)
#     return ks, evecs
# end

### Dependencies of MPB interface in case this should be a sub-module
# using PyCall
# using Distributed
# using GeometryPrimitives
# using OptiMode
# using HDF5
# using DelimitedFiles
# using EllipsisNotation
# using Printf
# using LinearAlgebra: diag
# using ProgressMeter
# using Tullio

# SHM3 = SHermitianCompact{3,Float64,6}
# Conda.add("wurlitzer"; channel="conda-forge")

const DEFAULT_MPB_LOGPATH = "log.txt"
const DEFAULT_MPB_EPSPATH = "-epsilon.h5"


# function __init__()
#     # @eval global mp = pyimport("meep")
#     # @eval global mpb = pyimport("meep.mpb")
#     # @eval global np = pyimport("numpy")
#     copy!(pymeep, pyimport("meep"))
#     copy!(pympb, pyimport("meep.mpb"))
#     # copy!(numpy, pyimport("numpy"))
#     # wurlitzer = pyimport("wurlitzer")
#     py"""
#     import numpy
#     import h5py
#     from numpy import linspace, transpose
#     from scipy.interpolate import interp2d
#     from meep import Medium, Vector3

#     def return_evec(evecs_out):
#         fn = lambda ms, band: numpy.copyto(evecs_out[:,:,band-1],ms.get_eigenvectors(band,1)[:,:,0])
#         return fn
        
#     def save_evecs(ms,band):
#         ms.save_eigenvectors(ms.filename_prefix + f"-evecs.b{band:02}.h5")
        
#     def return_and_save_evecs(evecs_out):
#         # fn = lambda ms, band: ms.save_eigenvectors(ms.filename_prefix + f"-evecs.b{band:02}.h5"); numpy.copyto(evecs_out[:,:,band-1],ms.get_eigenvectors(band,1)[:,:,0]) )
#         def fn(ms,band):
#             ms.save_eigenvectors(ms.filename_prefix + f"-evecs.b{band:02}.h5")
#             numpy.copyto(evecs_out[:,:,band-1],ms.get_eigenvectors(band,1)[:,:,0])
#         return fn

#     def output_dfield_energy(ms,band):
#         D = ms.get_dfield(band, bloch_phase=False)
#         # U, xr, xi, yr, yi, zr, zi = ms.compute_field_energy()
#         # numpy.savetxt(
#         #     f"dfield_energy.b{band:02}.csv",
#         #     [U, xr, xi, yr, yi, zr, zi],
#         #     delimiter=","
#         # )
#         ms.compute_field_energy()

#     # load epsilon data into python closure function `fmat(p)`
#     # `fmat(p)` should accept an input point `p` of type meep.Vector3 as a single argument
#     # and return a "meep.Material" object with dielectric tensor data for that point
#     def matfn_from_file(fpath,Dx,Dy,Nx,Ny):
#         x = linspace(-Dx/2., Dx*(0.5 - 1./Nx), Nx)
#         y = linspace(-Dy/2., Dy*(0.5 - 1./Ny), Ny)
#         with h5py.File(fpath, 'r') as f:
#             f_epsxx,f_epsxy,f_epsxz,f_epsyy,f_epsyz,f_epszz = [interp2d(x,y,transpose(f["epsilon."+s])) for s in ["xx","xy","xz","yy","yz","zz"] ]
#         matfn = lambda p : Medium(epsilon_diag=Vector3( f_epsxx(p.x,p.y)[0], f_epsyy(p.x,p.y)[0], f_epszz(p.x,p.y)[0] ),epsilon_offdiag=Vector3( f_epsxy(p.x,p.y)[0], f_epsxz(p.x,p.y)[0], f_epsyz(p.x,p.y)[0] ))
#         return matfn

#     # Transfer Julia epsilon data directly into python closure function `fmat(p)`
#     # `fmat(p)` should accept an input point `p` of type meep.Vector3 as a single argument
#     # and return a "meep.Material" object with dielectric tensor data for that point
#     def matfn(eps,x,y):
#         f_epsxx,f_epsxy,f_epsxz,f_epsyy,f_epsyz,f_epszz = [interp2d(x,y,transpose(eps[ix,iy,:,:])) for (ix,iy) in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)] ]
#         matfn = lambda p : Medium(epsilon_diag=Vector3( f_epsxx(p.x,p.y)[0], f_epsyy(p.x,p.y)[0], f_epszz(p.x,p.y)[0] ),epsilon_offdiag=Vector3( f_epsxy(p.x,p.y)[0], f_epsxz(p.x,p.y)[0], f_epsyz(p.x,p.y)[0] ))
#         return matfn
#     """
# end

function get_Δs_mpb(ms)
    Δx, Δy, Δz = ms.geometry_lattice.size.__array__()
end

function get_Ns_mpb(ms)
    Nx, Ny, Nz = ms._get_grid_size().__array__()
end

function get_xyz_mpb(ms)
    Δx, Δy, Δz = get_Δs_mpb(ms) # [ (Δ==0. ? 1. : Δ) for Δ in ms_size.__array__() ]
    Nx, Ny, Nz = get_Ns_mpb(ms)
    # x = ((Δx / Nx) .* (0:(Nx-1))) .- Δx / 2.0
    # y = ((Δy / Ny) .* (0:(Ny-1))) .- Δy / 2.0
    # z = ((Δz / Nz) .* (0:(Nz-1))) .- Δz / 2.0
    x = LinRange(-Δx/2, Δx/2 - Δx/Nx, Nx)
    y = LinRange(-Δy/2, Δy/2 - Δy/Ny, Ny)
    z = LinRange(-Δz/2, Δz/2 - Δz/Nz, Nz)
    return x, y, z
end

function get_ε⁻¹_mpb(ms)
    x, y, z = get_xyz_mpb(ms)
    Nx = length(x)
    Ny = length(y)
    Nz = length(z)
    ε⁻¹ = Array{Float64,5}(undef, (3, 3, Nx, Ny, Nz))
    for i = 1:Nx, j = 1:Ny, k = 1:Nz
        ε⁻¹[:, :, i, j, k] .= real(ms.get_epsilon_inverse_tensor_point(pymeep.Vector3(
            x[i],
            y[j],
            z[k],
        )).__array__())
    end
    return ε⁻¹
end

function get_ε_mpb(ms)
    x, y, z = get_xyz_mpb(ms)
    Nx = length(x)
    Ny = length(y)
    Nz = length(z)
    ε = Array{Float64,5}(undef, (3, 3, Nx, Ny, Nz))
    for i = 1:Nx, j = 1:Ny, k = 1:Nz
        ε[:, :, i, j, k] .= real(ms.get_epsilon_inverse_tensor_point(pymeep.Vector3(
            x[i],
            y[j],
            z[k],
        )).inverse().__array__())
    end
    return ε
end

mpMatrix(M::Matrix) = pymeep.Matrix([pymeep.Vector3(M[:,j]...) for j=1:size(M)[2]]...)

function mpMedium(mat,λ)
    eps = mat.fε(λ)
    return pymeep.Medium(epsilon_diag=pymeep.Vector3(eps[1,1],eps[2,2],eps[3,3]),epsilon_offdiag=pymeep.Vector3(eps[1,2],eps[1,3],eps[2,3]))
end

"""
load a set of eigenvectors saved by MPB

initial axis order: 
    band_idx=1:num_bands, mn_idx=1:2, G_idx=1:N(grid)
intermediate (permuted) axis order: 
    mn_idx=1:2, G_idx=1:N(grid), band_idx=1:num_bands
final (flattened) axis order:
    mn_G_idx=1:(2*N(grid)), band_idx=1:num_bands
"""
function load_evecs(fpath,grid::Grid)
    return h5open(fpath, "r") do file
        # Hv0 = permutedims( read(file, "rawdata") ,(2,3,1))
        # Ngrid = size(Hv0,2)
        # Nbands = size(Hv0,3)
        # return reshape(Hv0,(2*Ngrid,Nbands))
        Hv0 = read(file, "rawdata")
        Nbands = size(Hv0,1)
        Ngrid = N(grid)
        @assert isequal(size(Hv0,3), Ngrid)
        grid_size = size(grid)
        ND = ndims(grid)
        perm = (1,(reverse(1:ND).+1)...,ND+2)
        Hv1 =  copy( reshape( permutedims( Hv0, (2,3,1)), (2,reverse(grid_size)...,Nbands) ) )
        return reshape(permutedims(Hv1,perm),(2*Ngrid,Nbands))
    end
end

"""
load a single eigenvector saved by MPB

initial axis order: 
    band_idx=1:num_bands, mn_idx=1:2, G_idx=1:N(grid)
intermediate (permuted) axis order: 
    mn_idx=1:2, G_idx=1:N(grid), band_idx=1:num_bands
final (flattened) axis order:
    mn_G_idx=1:(2*N(grid)), band_idx=1:num_bands
"""
function load_evec(fpath,band_idx,grid::Grid)
    return h5open(fpath, "r") do file
        grid_size = size(grid)
        ND = ndims(grid)
        perm = (1,(reverse(1:ND).+1)...)
        Hv0 = reshape(read(file, "rawdata")[band_idx,:,:],(2,reverse(grid_size)...))
        return vec( permutedims(Hv0, perm ) )
    end
end

function load_evec_arr(fpath,band_idx,grid::Grid)
    return h5open(fpath, "r") do file
        grid_size = size(grid)
        ND = ndims(grid)
        perm = (1,(reverse(1:ND).+1)...)
        Hv0 = reshape(read(file, "rawdata")[band_idx,:,:],(2,reverse(grid_size)...))
        return permutedims(Hv0, perm )
    end
end

function load_epsilon(fpath)
    return h5open(fpath, "r") do file
        ε_ave0 = read(file, "data")
        perm = reverse(1:ndims(ε_ave0))
        ε_ave = permutedims(ε_ave0,perm)
        grid_size = size(ε_ave)

        ε_xx = permutedims( read(file, "epsilon.xx"), perm )
        ε_xy = permutedims( read(file, "epsilon.xy"), perm )
        ε_xz = permutedims( read(file, "epsilon.xz"), perm )
        ε_yy = permutedims( read(file, "epsilon.yy"), perm )
        ε_yz = permutedims( read(file, "epsilon.yz"), perm )
        ε_zz = permutedims( read(file, "epsilon.zz"), perm )
        ε⁻¹_xx = permutedims( read(file, "epsilon_inverse.xx"), perm )
        ε⁻¹_xy = permutedims( read(file, "epsilon_inverse.xy"), perm )
        ε⁻¹_xz = permutedims( read(file, "epsilon_inverse.xz"), perm )
        ε⁻¹_yy = permutedims( read(file, "epsilon_inverse.yy"), perm )
        ε⁻¹_yz = permutedims( read(file, "epsilon_inverse.yz"), perm )
        ε⁻¹_zz = permutedims( read(file, "epsilon_inverse.zz"), perm )
        
        # return data,ε_xx,ε_xy,ε_xz,ε_yy,ε_yz,ε_zz,ε⁻¹_xx,ε⁻¹_xy,ε⁻¹_xz,ε⁻¹_yy,ε⁻¹_yz,ε⁻¹_zz
        # ε = reshape(cat(reshape.((ε_xx,ε_xy,ε_xz,ε_xy,ε_yy,ε_yz,ε_xz,ε_yz,ε_zz),((1,grid_size...),))...,dims=1),(3,3,grid_size...))
        # ε⁻¹ = reshape(cat(reshape.((ε⁻¹_xx,ε⁻¹_xy,ε⁻¹_xz,ε⁻¹_xy,ε⁻¹_yy,ε⁻¹_yz,ε⁻¹_xz,ε⁻¹_yz,ε⁻¹_zz),((1,grid_size...),))...,dims=1),(3,3,grid_size...))
        ε = reshape(cat(reshape.((ε_xx,conj.(ε_xy),conj.(ε_xz),ε_xy,ε_yy,conj.(ε_yz),ε_xz,ε_yz,ε_zz),((1,grid_size...),))...,dims=1),(3,3,grid_size...))
        ε⁻¹ = reshape(cat(reshape.((ε⁻¹_xx,conj.(ε⁻¹_xy),conj.(ε⁻¹_xz),ε⁻¹_xy,ε⁻¹_yy,conj.(ε⁻¹_yz),ε⁻¹_xz,ε⁻¹_yz,ε⁻¹_zz),((1,grid_size...),))...,dims=1),(3,3,grid_size...))
        return ε, ε⁻¹, ε_ave
    end
end

function load_field(fpath)
    return h5open(fpath, "r") do file
        # desc_str = read(file,"description")
        # bw = read(file,"Bloch wavevector")
        f_xr0 = read(file, "x.r")
        perm = reverse(1:ndims(f_xr0))
        f_xr = permutedims(f_xr0,perm)
        grid_size = size(f_xr)

        f_xi = permutedims( read(file, "x.i"), perm )
        f_yr = permutedims( read(file, "y.r"), perm )
        f_yi = permutedims( read(file, "y.i"), perm )
        f_zr = permutedims( read(file, "z.r"), perm )
        f_zi = permutedims( read(file, "z.i"), perm )
        f = reshape(cat(reshape.((f_xr+1im*f_xi,f_yr+1im*f_yi,f_zr+1im*f_zi),((1,grid_size...),))...,dims=1),(3,grid_size...))
        return f
    end
end

function save_epsilon(fpath,ε,ε⁻¹,ε_ave)
    perm_ave = reverse(1:ndims(ε_ave))
    perm = (1,2,(perm_ave.+2)...)
    ε_avep = permutedims(ε_ave,perm_ave)
    εp = permutedims(ε, perm)
    ε⁻¹p = permutedims(ε⁻¹, perm)
    h5open(fpath, "w") do file
        write(file, "data", ε_avep)
        write(file, "epsilon.xx", εp[1,1,..])
        write(file, "epsilon.xy", εp[1,2,..])
        write(file, "epsilon.xz", εp[1,3,..])
        write(file, "epsilon.yy", εp[2,2,..])
        write(file, "epsilon.yz", εp[2,3,..])
        write(file, "epsilon.zz", εp[3,3,..])
        write(file, "epsilon_inverse.xx", ε⁻¹p[1,1,..])
        write(file, "epsilon_inverse.xy", ε⁻¹p[1,2,..])
        write(file, "epsilon_inverse.xz", ε⁻¹p[1,3,..])
        write(file, "epsilon_inverse.yy", ε⁻¹p[2,2,..])
        write(file, "epsilon_inverse.yz", ε⁻¹p[2,3,..])
        write(file, "epsilon_inverse.zz", ε⁻¹p[3,3,..])
    end
end

function save_epsilon(fpath,ε,ε⁻¹)
    ε_ave =  ( view(ε,1,1,..) .+ view(ε,2,2,..) .+ view(ε,3,3,..) ) ./ 3.0
    save_epsilon(fpath,ε,ε⁻¹,ε_ave)
end

function save_epsilon(fpath,ε)
    ε⁻¹ = copy(ε)
    for i in CartesianIndices(size(ε)[3:end]) #eachindex(view(ε,1,1,..))
        ε⁻¹[:,:,i] =  inv(ε[:,:,i])
    end
    save_epsilon(fpath,ε,ε⁻¹)
end

# Progress bar utility functions
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
prog_interactive(dt) = Progress(dt; output = stderr, enabled = !is_logging(stderr))

# Parsing functions for log file
function findlines(fpath,str::AbstractString)
    return open(fpath,"r") do file
        filter(line->occursin(str,line),readlines(file))
    end
end

function findlines(fpath,strs::AbstractVector{TS}) where {TS<:AbstractString}
    return open(fpath,"r") do file
        match_lines = filter(
            line->mapreduce(s->occursin(s,line),|,strs),
            readlines(file),
        )
        return map(s->filter(line->occursin(s,line),match_lines),strs)
    end
end

function findlines(fpath,line_filt::TF) where {TF<:Function}
    return open(fpath,"r") do file
        filter(line_filt,readlines(file))
    end
end

"""
from the mpb docs:
https://pympb.readthedocs.io/en/latest/Python_User_Interface/#the-inverse-problem-k-as-a-function-of-frequency

we expect the kvals line format to be
"kvals: omega, band-min, band-max, korig1, korig2, korig3, kdir1, kdir2, kdir3, k magnitudes... "
"""
function parse_kvals(kvals_line)
    vals = parse.((Float64,),strip.(split(kvals_line,","))[2:end])
    omega, band_min, band_max, korig1, korig2, korig3, kdir1, kdir2, kdir3 = vals[1:9]
    kmags = vals[10:end]
    return omega, band_min, band_max, korig1, korig2, korig3, kdir1, kdir2, kdir3, kmags
end

function parse_kvecs(kvals_line)
    omega, band_min, band_max, korig1, korig2, korig3, kdir1, kdir2, kdir3, kmags = parse_kvals(kvals_line)
    kdir = SVector{3,Float64}(kdir1, kdir2, kdir3)
    kvecs =  (km*kdir for km in kmags)
    return kvecs
end

parse_kmags(kvals_line) = parse_kvals(kvals_line)[10]

function parse_neffs(kvals_line)
    omega, band_min, band_max, korig1, korig2, korig3, kdir1, kdir2, kdir3, kmags = parse_kvals(kvals_line)
    return kmags./omega
end

"""
group velocity line format example:
"velocity:, 1, Vector3<-8.863934120552942e-06, -2.9413768567368583e-06, 0.41192396407478216>"

see the mpb docs:
https://pympb.readthedocs.io/en/latest/Python_User_Interface/#miscellaneous-functions
"""
function parse_grpvels(grpvel_line)
    gv_strs = split(grpvel_line,"Vector3")[2:end]
    gv_vecs = ( parse.((Float64,),strip.(split(strip(gv_str,['<','>',' ',',']),","))) for gv_str in gv_strs )
    return collect(gv_vecs)
end

parse_gv_vecs(grpvel_line) = parse_grpvels(grpvel_line)

parse_gv_mags(grpvel_line;kdir=[0.,0.,1.]) = dot.(parse_gv_vecs(grpvel_line),(kdir,))

parse_ngs(grpvel_line;kdir=[0.,0.,1.]) = inv.(parse_gv_mags(grpvel_line;kdir))

"""
f-energy-components line format example:
"D-energy-components:, 1, 1, 0.00358093, 0.82414, 0.172279"

from the mpb docs:
https://pympb.readthedocs.io/en/latest/Python_User_Interface/#loading-and-manipulating-the-current-field

we expect the `f`-energy line format to be
"`f`-energy-components:, k-index, band-index, x-fraction, y-fraction, z-fraction"
where `f` can be either `D` or `H`
"""
function parse_f_energy_components(f_energy_line::AbstractString)
    k_idx_str, band_idx_str, x_frac_str, y_frac_str, z_frac_str = strip.(split(f_energy_line,",")[2:end])
    x_frac, y_frac, z_frac = parse.((Float64,), (x_frac_str, y_frac_str, z_frac_str) )
    k_idx, band_idx = parse.((Int64,), (k_idx_str, band_idx_str) )
    return k_idx, band_idx, x_frac, y_frac, z_frac
end

function parse_f_energy_components(f_energy_lines::AbstractVector)
    k_idx, band_idx, x_frac, y_frac, z_frac = ( getindex.(parse_f_energy_components.(f_energy_lines),(ii,)) for ii in 1:5 )
    return k_idx, band_idx, x_frac, y_frac, z_frac
end

log_fname(filename_prefix) = lstrip(join((filename_prefix,"log"),"."),'.')
log_fpath(filename_prefix;data_path=pwd()) = joinpath(data_path,log_fname(filename_prefix))

function parse_ω_k(filename_prefix;data_path=pwd(),kdir=[0.0,0.0,1.0])
    k_lines = findlines(log_fpath(filename_prefix;data_path),["kvals",])[1]
    omega, band_min, band_max, korig1, korig2, korig3, kdir1, kdir2, kdir3, kmags = parse_kvals(k_lines[1])
    @assert kdir ≈ [kdir1, kdir2, kdir3]
    return omega, kmags
end

function parse_findk_log(filename_prefix="f01";data_path=pwd())
    # k_lines, gv_lines, d_energy_lines = findlines(log_fname(filename_prefix),["kvals","velocity","D-energy"])
    println(joinpath(data_path,log_fname(filename_prefix)))
    k_lines, gv_lines, d_energy_lines = findlines(joinpath(data_path,log_fname(filename_prefix)),["kvals","velocity","D-energy"])
    omega, band_min, band_max, korig1, korig2, korig3, kdir1, kdir2, kdir3, kmags = parse_kvals(k_lines[1])
    kdir = SVector{3,Float64}(kdir1, kdir2, kdir3)
    neffs = kmags./omega
    ngs = [parse_ngs(gv_line;kdir)[igvl] for (igvl,gv_line) in enumerate(gv_lines)]
    k_idx, band_idx, x_frac, y_frac, z_frac = parse_f_energy_components(d_energy_lines)
    return omega, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac
end

function parse_findk_logs(ωinds;data_path=pwd(),pfx="")
    # parsed_logs = (parse_findk_log(;logpath= joinpath(data_path,(@sprintf "log.f%02i.txt" ωind))) for ωind in ωinds)
    parsed_logs = (parse_findk_log((lstrip(join([pfx,(@sprintf "f%02i" ωind)],'.'),'.')) ;data_path) for ωind in ωinds)
    omega, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac = ( getindex.(parsed_logs,(ii,)) for ii in 1:8 )
    return omega, kdir, neffs, ngs, band_idx, x_frac, y_frac, z_frac
end



res_mpb(grid::Grid{2}) = pymeep.Vector3((grid.Nx / grid.Δx), (grid.Ny / grid.Δy), 1)
res_mpb(grid::Grid{3}) = pymeep.Vector3((grid.Nx / grid.Δx), (grid.Ny / grid.Δy), (grid.Nz / grid.Δz))
lat_mpb(grid::Grid{2}) = pymeep.Lattice(size = pymeep.Vector3(grid.Δx, grid.Δy, 1.0)) #pymeep.Lattice(size = pymeep.Vector3(grid.Δx, grid.Δy, 0))
lat_mpb(grid::Grid{3}) = pymeep.Lattice(size = pymeep.Vector3(grid.Δx, grid.Δy, grid.Δz))


function mpGeomObj(b::GeometryPrimitives.Box{2};λ=1.55,thickness=10.0)
    return pymeep.Block(
        center = pymeep.Vector3(b.c...),
        size = 2 * pymeep.Vector3(b.r[1],b.r[2],thickness),
        e1 = pymeep.Vector3(b.p[:,1]...),
        e2 = pymeep.Vector3(b.p[:,2]...),
        e3 = pymeep.Vector3(0,0,1),
        material=mpMedium(b.data,λ),
        )
end

function mpGeomObj(b::GeometryPrimitives.Box{3};λ=1.55)
    return pymeep.Block(
        center = pymeep.Vector3(b.c...),
        size = 2 * pymeep.Vector3(b.r...),
        e1 = pymeep.Vector3(b.p[:,1]...),
        e2 = pymeep.Vector3(b.p[:,2]...),
        e3 = pymeep.Vector3(b.p[:,3]...),
        material=mpMedium(b.data,λ),
        )
end

function mpGeomObj(p::GeometryPrimitives.Polygon;λ=1.55,thickness=10.0)
    return pymeep.Prism(
        [pymeep.Vector3(p.v[i,1],p.v[i,2],-thickness/2.) for i=1:size(p.v)[1]],
        height = thickness,
        material=mpMedium(p.data,λ),
        )
end

mpGeom(shapes::AbstractVector;λ=1.55) = mpGeomObj.(shapes;λ)
mpGeom(geom::OptiMode.Geometry;λ=1.55) = mpGeomObj.(geom.shapes;λ)

function ms_mpb(;resolution=10,
    is_negative_epsilon_ok=false,
    eigensolver_flops=0,
    # is_eigensolver_davidson=false,
    eigensolver_nwork=3,
    eigensolver_block_size=-11,                     # TODO: remember what negative block size means
    eigensolver_flags=68,
    use_simple_preconditioner=false,
    force_mu=false,
    mu_input_file="",
    epsilon_input_file="",                          # always empty, we feed in pre-smoothed `ε` as `default_material`
    mesh_size=3,                                    # auto set via `grid`, see below
    target_freq=0.0,
    tolerance=1.0e-7,
    num_bands=1,                                    # set in solve_ω or solve_k via `nev`
    k_points=[],
    ensure_periodicity=true,
    geometry=[],                                    # always empty, we feed in pre-smoothed `ε` as `default_material`
    geometry_lattice=pymeep.Lattice(),              # auto set via `grid`, see below
    geometry_center=pymeep.Vector3(0, 0, 0),
    default_material=pymeep.Medium(epsilon=1),      # normally set to my pre-smoothed `ε` data (to avoid further smoothing)
    dimensions=3,                                   # legacy input field (AFAIU), now inferred from `ε` data format
    random_fields=false,
    filename_prefix="",                             # passed via equivalent `find_k` keyword arg
    deterministic=true,
    verbose=false,
    parity=pymeep.NO_PARITY)                        # should start to use this at some point...
    ms = pympb.ModeSolver(;
        resolution,
        is_negative_epsilon_ok,
        eigensolver_flops,
        # is_eigensolver_davidson,
        eigensolver_nwork,
        eigensolver_block_size,
        eigensolver_flags,
        use_simple_preconditioner,
        force_mu,
        mu_input_file,
        epsilon_input_file,
        mesh_size,
        target_freq,
        tolerance,
        num_bands,
        k_points,
        ensure_periodicity,
        geometry,
        geometry_lattice,
        geometry_center,
        default_material,
        dimensions,
        random_fields,
        filename_prefix,
        deterministic,
        verbose,
    )
    ms.init_params(parity, false)   # second argument is "reset_fields"
    return ms
end

function ms_mpb(grid::Grid{ND};kwargs...) where ND
    return ms_mpb(; kwargs...,
        geometry_lattice = lat_mpb(grid),
        resolution = res_mpb(grid),
        # dimensions = ND,
    )
end

function ms_mpb(geom::OptiMode.Geometry,grid::Grid{ND};λ=1.55,kwargs...) where ND
    return ms_mpb(; kwargs...,
        geometry_lattice = lat_mpb(grid),
        geometry = mpGeom(geom;λ),
        resolution = res_mpb(grid),
        # dimensions = ND,
    )
end

function ms_mpb(eps::AbstractArray,grid::Grid{ND};kwargs...) where ND
    return ms_mpb(; kwargs...,
        mesh_size=1,
        geometry_lattice = lat_mpb(grid),
        resolution = res_mpb(grid),
        default_material = py"matfn"(eps,x(grid),y(grid))
        # dimensions = ND,
    )
end

function n_range(geom::OptiMode.Geometry;λ=1.55)
    return extrema(vcat([sqrt.(diag(eps)) for eps in map.(geom.fεs,(λ,))]...))
end

function n_range(ε::AbstractArray)
    return sqrt.(extrema(vcat( (vec(view(ε,i,i,..)) for i=1:3)... ) ))
end

function n_range(ε::AbstractVector)
    return sqrt.(extrema(vcat((view(ee,i,i,..) for ee in ε, i=1:3)...)))
end

function rename_findk_data(filename_prefix,b_inds;k_ind=1,data_path=pwd(),overwrite=false)
    fnames = readdir(data_path)
    for b_ind in b_inds
        evecs_source_fname = filename_prefix * (@sprintf "-evecs.b%02i.h5" b_ind)
        if in(evecs_source_fname,fnames)
            evecs_target_fname = "evecs." * filename_prefix * (@sprintf ".b%02i.h5" b_ind)
            mv(
                joinpath(data_path,evecs_source_fname),
                joinpath(data_path,evecs_target_fname);
                force=overwrite,
            )
        end
        efield_source_fname = filename_prefix * (@sprintf "-e.k%02i.b%02i.h5" k_ind b_ind)
        if in(efield_source_fname,fnames)
            efield_target_fname = "e." * filename_prefix * (@sprintf ".b%02i.h5" b_ind)
            mv(
                joinpath(data_path,efield_source_fname),
                joinpath(data_path,efield_target_fname);
                force=overwrite,
            )
        end
        hfield_source_fname = filename_prefix * (@sprintf "-h.k%02i.b%02i.h5" k_ind b_ind)
        if in(hfield_source_fname,fnames)
            hfield_target_fname = "h." * filename_prefix * (@sprintf ".b%02i.h5" b_ind)
            mv(
                joinpath(data_path,hfield_source_fname),
                joinpath(data_path,hfield_target_fname);
                force=overwrite,
            )
        end
    end
    # log_source_fname = logpath
    # log_target_fname = @sprintf "log.f%02i.txt" f_ind
    # mv(
    #     joinpath(data_path,log_source_fname),
    #     joinpath(data_path,log_target_fname);
    #     force=overwrite,
    # )
    eps_source_fname = filename_prefix * "-epsilon.h5"
    if in(eps_source_fname,fnames)
        eps_target_fname = "eps." * filename_prefix * ".h5" 
        mv(
            joinpath(data_path,eps_source_fname),
            joinpath(data_path,eps_target_fname);
            force=overwrite,
        )
    end
end

k_fname(filename_prefix) = join(("k",filename_prefix,"csv"),".")
write_k(kmags,omega,filename_prefix;data_path=pwd()) = writedlm(joinpath(data_path,k_fname(filename_prefix)),hcat(kmags,omega*ones(length(kmags))),',')
read_k(filename_prefix;data_path=pwd()) = ( k_om=readdlm(joinpath(data_path,k_fname(filename_prefix)),','); return (copy(k_om[:,1]),k_om[1,2]) )

function already_calculated(filename_prefix,bands;data_path=pwd())::Bool
    fnames = readdir(data_path)
    evecs_fnames = ( "evecs." * filename_prefix * (@sprintf ".b%02i.h5" bidx) for bidx in bands)
    # log_fname = lstrip(join((filename_prefix,"log"),"."),'.')         # TODO: fix logging (stopped working when I started multiprocessing) 
    # all(x->in(x,fnames),evecs_fnames) && in(log_fname,fnames)         # Currently C stdout piping gets randomly mixed from different worker procs, so no log files saved. 
    all(x->in(x,fnames),evecs_fnames) 
end

function already_calculated(ω,filename_prefix,bands;data_path=pwd())::Bool
    already_calculated(filename_prefix,bands;data_path) && isapprox(read_k(filename_prefix;data_path)[2],ω)
end


function _find_k(ω::Real,ε::AbstractArray,grid::Grid{ND};num_bands=2,band_min=1,band_max=num_bands,eig_tol=1e-8,k_tol=1e-8,
    k_dir=[0.,0.,1.],parity=pymeep.NO_PARITY,n_guess_factor=0.9,data_path=pwd(),filename_prefix="f01",overwrite=false,
    save_evecs=true,save_efield=false,save_hfield=false,calc_polarization=false,calc_grp_vels=false,
    band_func=[pympb.fix_efield_phase],kwargs...) where ND
    # Create the set of functions applied to each solution (`band_func`)
    n_bands_out = band_max-band_min+1
    evecs = PyReverseDims(zeros(ComplexF64,(N(grid),2,n_bands_out))) # zeros(ComplexF64,(N(grid),2,n_bands_out))
    if save_evecs
        push!(band_func,py"return_and_save_evecs"(evecs))
    else
        push!(band_func,py"return_evecs"(evecs))
    end
    save_efield && push!(band_func,pympb.output_efield)
    save_hfield && push!(band_func,pympb.output_hfield)
    calc_polarization && push!(band_func,py"output_dfield_energy")
    calc_grp_vels && push!(band_func,pympb.display_group_velocities)
    # Move into directory where output data will be stored
    init_path = pwd()
    cd(data_path)
    n_min, n_max = n_range(ε)
    n_guess = n_guess_factor * n_max
    kmags = open(log_fname(filename_prefix),"w") do logfile
        return redirect_stdout(logfile) do
            ms = ms_mpb(                    # create a pympb.modesolver object
                ε,
                grid;
                filename_prefix,
                num_bands,
                tolerance=eig_tol,
                kwargs...,
            )
            ms.output_epsilon()             # save ε data in working directory
            kmags = ms.find_k(              # run `pympb.solve_k`,  input variable names & meanings:
                parity,                     #   `parity`:         mode field parity constraints (one or more meep parity objects)
                ω,                          #   `ω`:              frequency at which to solve for |k|
                band_min,                   #   `band_min`:       find |k|(ω) for bands `band_min`:`band_max`
                band_max,                   #   `band_max`:       see previous
                pymeep.Vector3(k_dir...),   #   `k`:              k⃗ direction in which to search
                k_tol,                      #   `tol`:            |k| (absolute?) error tolerance   #   TODO: check if this is absolute or relative tolerance
                n_guess * ω,                #   `kmag_guess`:,    |k| estimate
                n_min * ω,                  #   `kmag_min`:       find |k| in range `kmag_min`:`kmag_max`
                n_max * ω,                  #   `kmag_max`:       see previous
                band_func...                #   `*band_func`:     set of functions applied to each (|k|,eigenvector) solution 
            )
            return kmags
        end
    end
    rename_findk_data(filename_prefix,band_min:band_max;data_path,overwrite)
    write_k(kmags,ω,filename_prefix)
    # evecs   =   [load_evec_arr( joinpath(data_path,("evecs." * filename_prefix * (@sprintf ".b%02i.h5" bidx))), bidx, grid) for bidx=band_min:band_max]
    
    #### reshuffle spatial grid axes of returned mpb eigenvector data.... kind of confusing ####
    # I think the axis order of the returned eigenvector data, once passed through PyCall, is
    #       [ band_idx=1:num_bands, mn_idx=1:2, G_idx=1:N(grid) ]
    # but reshaping the 1:N(grid) axis to mulitple axes 1:Nx,1:Ny(,1:Nz) requires reversing the grid axis order.
    # ie. it starts out as (Nz x Ny x Nx), and here we swap those around to be (Nx x Ny x Nz) 
    # before re-flattening each band solution as a vector (an eigenvector of our HelmholtzOperator)
    evecs_out = vec.(copy.(collect(eachslice(permutedims(reshape(PyArray(evecs),(num_bands,2,reverse(size(grid))...)),(1,2,(reverse(1:ND).+2)...)),dims=1))))
    
    cd(init_path)
    return kmags, evecs_out
end

# function find_k(ω::Real,ε::AbstractArray,grid::Grid{ND};k_dir=[0.,0.,1.], num_bands=2,band_min=1,band_max=num_bands,filename_prefix="f01",
#     band_func=[pympb.fix_efield_phase,py"output_evecs"],save_efield=false,save_hfield=false,calc_polarization=false,calc_grp_vels=false,
#     parity=pymeep.NO_PARITY,n_guess_factor=0.9,data_path=pwd(),kwargs...) where ND  

function find_k(ω::Real,ε::AbstractArray,grid::Grid{ND};num_bands=2,band_min=1,band_max=num_bands,filename_prefix="f01",data_path=pwd(),overwrite=false,kwargs...) where ND  
    if !overwrite && already_calculated(ω,filename_prefix,band_min:band_max;data_path)
        println("loading pre-calculated data for "*filename_prefix)
        kmags = read_k(filename_prefix;data_path)[1]
        evecs = [load_evec_arr( joinpath(data_path,("evecs." * filename_prefix * (@sprintf ".b%02i.h5" bidx))), bidx, grid) for bidx=band_min:band_max]
    else
        kmags,evecs = _find_k(ω,ε,grid;filename_prefix,num_bands,band_min,band_max,data_path,overwrite,kwargs...)
    end
    return kmags,evecs
end

function find_k(ω::AbstractVector,ε::AbstractVector,grid::Grid{ND};worker_pool=default_worker_pool(),filename_prefix="",data_path=pwd(),kwargs...) where ND  
    nω = length(ω)
    prefixes = [ lstrip(join((filename_prefix,(@sprintf "f%02i" fidx)),"."),'.') for fidx=1:nω ]
    k_evec = progress_pmap(worker_pool,ω,ε,prefixes) do om, eps, prfx
        return find_k(om,eps,grid;filename_prefix=prfx,data_path,kwargs...)
    end
    ks = mapreduce(x->getindex(x,1),hcat,k_evec) |> transpose 
    evecs = permutedims(mapreduce(x->getindex(x,2),hcat,k_evec),(2,1))
    return ks,evecs
end

function find_k(ω::Real,p::AbstractVector,geom_fn::TF,grid::Grid{ND};kwargs...) where {ND,TF<:Function}
    eps = copy(smooth(ω,p,:fεs,false,geom_fn,grid).data)
    return find_k(ω,eps,grid;kwargs...)
end

function find_k(ω::AbstractVector,p::AbstractVector,geom_fn::TF,grid::Grid{ND}; kwargs...) where {ND,TF<:Function}
    eps = [copy(smooth(oo,p,:fεs,false,geom_fn,grid).data) for oo in ω]
    return find_k(ω,eps,grid; kwargs...)
end

# function group_index(k::Real,evec,om,ε⁻¹,∂ε∂ω,grid)
#     mag,mn = mag_mn(k,grid)
#     om / HMₖH(vec(evec),ε⁻¹,mag,mn) * (1-(om/2)*HMH(evec, _dot( ε⁻¹, ∂ε∂ω, ε⁻¹ ),mag,mn))
# end

# function group_index(ks::AbstractVector,evecs,om::Real,ε⁻¹,∂ε∂ω,grid)
#     [ group_index(ks[bidx],evecs[bidx],om,ε⁻¹,∂ε∂ω,grid) for bidx=1:num_bands ]
# end

# function group_index(ks::AbstractMatrix,evecs,om,ε⁻¹,∂ε∂ω,grid)
#     [ group_index(ks[fidx,bidx],evecs[fidx,bidx],om[fidx],ε⁻¹[fidx],∂ε∂ω[fidx],grid) for fidx=1:nω,bidx=1:num_bands ]
# end

# function group_index(ω::Real,p::AbstractVector,geom_fn::TF,grid::Grid{ND}; kwargs...) where {ND,TF<:Function}
#     eps, epsi = copy.(getproperty.(smooth(ω,p,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing),(:data,)))
#     deps_dom = ForwardDiff.derivative(oo->copy(getproperty(smooth(oo,p,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing)[1],:data)),ω)
#     k,evec = find_k(ω,eps,grid; kwargs...)
#     return group_index(k,evec,ω,epsi,deps_dom,grid)
# end

# function group_index(ω::AbstractVector,p::AbstractVector,geom_fn::TF,grid::Grid{ND}; worker_pool=default_worker_pool(),filename_prefix="",data_path=pwd(), kwargs...) where {ND,TF<:Function}
#     nω = length(ω)
#     prefixes = [ lstrip(join((filename_prefix,(@sprintf "f%02i" fidx)),"."),'.') for fidx=1:nω ]
#     eps_epsi = [ copy.(getproperty.(smooth(om,p,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing),(:data,))) for om in ω ]
#     deps_dom = [ ForwardDiff.derivative(oo->copy(getproperty(smooth(oo,p,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing)[1],:data)),om) for om in ω ]
#     ngs = progress_pmap(worker_pool,ω,eps_epsi,deps_dom,prefixes) do om, e_ei, de_do, prfx
#         kmags,evecs= find_k(om,e_ei[1],grid; filename_prefix=prfx, data_path, kwargs...)
#         return group_index(kmags,evecs,om,e_ei[2],de_do,grid)
#     end
#     return transpose(hcat(ngs...))
# end

function find_k_dom(ω::AbstractVector,p::AbstractVector,geom_fn::TF,grid::Grid{ND};dom=1e-4,data_path=pwd(),kwargs...) where {ND,TF<:Function}

    ks,evecs = find_k(ω,p,geom_fn,grid;data_path,kwargs...)

    dfp_path = joinpath(data_path,"dfp")
    mkpath(dfp_path)
    find_k(ω.+dom,p,geom_fn,grid;data_path=dfp_path,kwargs...)

    dfn_path = joinpath(data_path,"dfn")
    mkpath(dfn_path)
    find_k(ω.-dom,p,geom_fn,grid;data_path=dfn_path,kwargs...)

    return ks,evecs
end

function find_k_dom_dp(ω::AbstractVector,p::AbstractVector,geom_fn::TF,grid::Grid{ND};dp=1e-4*ones(length(p)),dom=1e-4,data_path=pwd(),kwargs...) where {ND,TF<:Function}
    
    ks,evecs = find_k_dom(ω,p,geom_fn,grid;dom,data_path,kwargs...)
    num_p = length(p)
    for pidx=1:num_p
        ddpp = [ (isequal(ii,pidx) ? dp[ii] : 0.0) for ii=1:num_p ]
        dpp_path = joinpath(data_path,(@sprintf "dpp%02i" pidx))
        mkpath(dpp_path)
        find_k_dom(ω,p.+ddpp,geom_fn,grid;dom,data_path=dpp_path,kwargs...)

        dpn_path = joinpath(data_path,(@sprintf "dpn%02i" pidx))
        mkpath(dpn_path)
        find_k_dom(ω,p.-ddpp,geom_fn,grid;dom,data_path=dpn_path,kwargs...)
    end
    return ks,evecs
end

