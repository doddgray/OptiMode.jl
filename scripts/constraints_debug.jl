# Parity operators based on MPB code:
# https://github.com/NanoComp/mpb/blob/master/src/maxwell/maxwell_constraints.c
using LoopVectorization
function _𝓟x!(Ho::AbstractArray{Complex{T},3},Hi::AbstractArray{Complex{T},3},grid::Grid{2}) where T<:Real
    Nx,Ny = size(grid)
	myzero = zero(eltype(Ho))
	temp1 = zero(eltype(Ho))
	temp2 = zero(eltype(Ho))
    @avx for iy ∈ 1:Ny
		# Ho[1,1,iy] = -Hi[1,1,iy]
		# Ho[2,1,iy] = myzero #zero(Complex{T})
		# Ho[1,1,iy] = Hi[1,1,iy] #myzero
		# Ho[2,1,iy] = Hi[2,1,iy] #zero(Complex{T})
		for ix ∈ 1:(Nx÷2)-1, l in 0:0 #1:((Nx÷2)-1), l in 0:0
			temp1 		= 0.5*( Hi[1+l,ix+1,iy] - Hi[1+l,Nx-ix+1,iy] )
			temp2 		= 0.5*( Hi[2+l,ix+1,iy] + Hi[2+l,Nx-ix+1,iy] )
			Ho[1+l,ix+1,iy] 	= temp1
			Ho[2+l,ix+1,iy] 	= temp2
			Ho[1+l,Nx-ix+1,iy] = -temp1
			Ho[2+l,Nx-ix+1,iy] = temp2
		end
	end
	return Ho
end

function _𝓟x̄!(Ho::AbstractArray{Complex{T},3},Hi::AbstractArray{Complex{T},3},grid::Grid{2}) where T<:Real
    Nx,Ny = size(grid)
	temp1 = zero(eltype(Ho))
	temp2 = zero(eltype(Ho))
	myzero = zero(eltype(Ho))
    @avx for iy ∈ 1:Ny
		# Ho[1,1,iy] = Hi[1,1,iy]
		# Ho[2,1,iy] = Hi[2,1,iy] #myzero #zero(Complex{T})
		# Ho[1,1,iy] = myzero
		# Ho[2,1,iy] = -Hi[2,1,iy] #zero(Complex{T})
		for ix ∈ 1:(Nx÷2)-1, l in 0:0 # 1:((Nx÷2)-1), l in 0:0
			temp1 		= 0.5*( Hi[1+l,ix+1,iy] + Hi[1+l,Nx-ix+1,iy] )
			temp2 		= 0.5*( Hi[2+l,ix+1,iy] - Hi[2+l,Nx-ix+1,iy] )
			Ho[1+l,ix+1,iy] 	= temp1
			Ho[2+l,ix+1,iy] 	= temp2
			Ho[1+l,Nx-ix+1,iy] = temp1
			Ho[2+l,Nx-ix+1,iy] = -temp2
		end
	end
	return Ho
end
𝓟x!(grid::Grid{2}) = LinearMap{ComplexF64}((Ho,Hi) -> vec(_𝓟x!(reshape(Ho,(2,grid.Nx,grid.Ny)),reshape(Hi,(2,grid.Nx,grid.Ny)),grid)),*(2,grid.Nx,grid.Ny),ishermitian=false,ismutating=true)
𝓟x!(grid::Grid{3}) = LinearMap{ComplexF64}((Ho,Hi) -> vec(_𝓟x!(reshape(Ho,(2,grid.Nx,grid.Ny,grid.Nz)),reshape(Hi,(2,grid.Nx,grid.Ny,grid.Nz)),grid)),*(2,grid.Nx,grid.Ny,grid.Nz),ishermitian=false,ismutating=true)
𝓟x̄!(grid::Grid{2}) = LinearMap{ComplexF64}((Ho,Hi) -> vec(_𝓟x̄!(reshape(Ho,(2,grid.Nx,grid.Ny)),reshape(Hi,(2,grid.Nx,grid.Ny)),grid)),*(2,grid.Nx,grid.Ny),ishermitian=false,ismutating=true)
𝓟x̄!(grid::Grid{3}) = LinearMap{ComplexF64}((Ho,Hi) -> vec(_𝓟x̄!(reshape(Ho,(2,grid.Nx,grid.Ny,grid.Nz)),reshape(Hi,(2,grid.Nx,grid.Ny,grid.Nz)),grid)),*(2,grid.Nx,grid.Ny,grid.Nz),ishermitian=false,ismutating=true)

##

eigind=1
geom = rwg(p)
Hr = reshape(HH[:,eigind],(2,Ns...));
E0 = E⃗(k,Hr,ω,geom,grid; svecs=false, normalized=false);
H0 = H⃗(k,Hr,ω,geom,grid; svecs=false, normalized=false);

HrPp = copy(Hr)
_𝓟x!(HrPp,copy(Hr),ms.grid);
# HrPp = reshape(HPp,(2,Ns...)); similar(Hr);
HrPn = copy(Hr)
_𝓟x̄!(HrPn,copy(Hr),ms.grid);
# HrPn = reshape(HPp,(2,Ns...)); similar(Hr);

H⃗1[] = H⃗(k1,Hr,ω,geom,grid; svecs=false, normalized=false)
Hx1[] = real(H⃗1[][1,:,:])
Hy1[] = real(H⃗1[][2,:,:])


H⃗Pp[] = H⃗(k1,HrPp,ω,geom,grid; svecs=false, normalized=false)
HxPp[] = real(H⃗Pp[][1,:,:])
HyPp[] = real(H⃗Pp[][2,:,:])

H⃗Pn[] = H⃗(k1,HrPn,ω,geom,grid; svecs=false, normalized=false)
HxPn[] = real(H⃗Pn[][1,:,:])
HyPn[] = real(H⃗Pn[][2,:,:])

E⃗1[] = E⃗(k1,Hr,ω,geom,grid; svecs=false, normalized=false)
Ex1[] = real(E⃗1[][1,:,:])
Ey1[] = real(E⃗1[][2,:,:])

E⃗Pp[] = E⃗(k1,HrPp,ω,geom,grid; svecs=false, normalized=false)
ExPp[] = real(E⃗Pp[][1,:,:])
EyPp[] = real(E⃗Pp[][2,:,:])

E⃗Pn[] = E⃗(k1,HrPn,ω,geom,grid; svecs=false, normalized=false)
ExPn[] = real(E⃗Pn[][1,:,:])
EyPn[] = real(E⃗Pn[][2,:,:])

Hs = reshape([  Hx1,  Hy1, HxPp ,HyPp ,  HxPn ,HyPn ],(2,3))
Es = reshape([  Ex1,  Ey1, ExPp ,EyPp ,  ExPn ,EyPn ],(2,3))
Hmagmax = [lift(A->maximum(abs,A),X) for X in Hs]
Emagmax = [lift(A->maximum(abs,A),X) for X in Es]

fig = Figure()
xs = x(ms.grid)
ys = y(ms.grid)

pos_H = [fig[i,j] for i=1:2,j=1:3]
ax_H = [Axis(pos[1,1]) for pos in pos_H]
pos_E = [fig[i,j] for i=3:4,j=1:3]
ax_E = [Axis(pos[1,1]) for pos in pos_E]

# nx = sqrt.(getindex.(inv.(ms.M̂.ε⁻¹),1,1))
cmaps_H = [:diverging_bwr_40_95_c42_n256, :diverging_protanopic_deuteranopic_bwy_60_95_c32_n256]
cmaps_E = [:diverging_bkr_55_10_c35_n256, :diverging_bky_60_10_c30_n256]
label_base = reshape( ["x","y","xPp","yPp","xPn","yPn",], (2,3))
labels_H = "H".*label_base
labels_E = "E".*label_base
heatmaps_H = [heatmap!(ax_H[i,j], xs, ys, Hs[i,j],colormap=cmaps_H[i],label=labels_H[i,j],colorrange=(-to_value(Hmagmax[i,j]),to_value(Hmagmax[i,j]))) for i=1:2,j=1:3]
heatmaps_E = [heatmap!(ax_E[i,j], xs, ys, Es[i,j],colormap=cmaps_E[i],label=labels_E[i,j],colorrange=(-to_value(Emagmax[i,j]),to_value(Emagmax[i,j]))) for i=1:2,j=1:3]
cbars_H = [Colorbar(pos_H[i,j][1, 2], heatmaps_H[i,j],  width=20 ) for i=1:2,j=1:3]
cbars_E = [Colorbar(pos_E[i,j][1, 2], heatmaps_E[i,j],  width=20 ) for i=1:2,j=1:3]

map( (axx,ll)->text!(axx,ll,position=(-1.4,1.1),textsize=0.7,color=:black), ax_H, labels_H )
map( (axx,ll)->text!(axx,ll,position=(-1.4,1.1),textsize=0.7,color=:white), ax_E, labels_E )
ax_all = vcat(ax_H,ax_E)
hidexdecorations!.(ax_all[1:end-1,:])
hideydecorations!.(ax_all[:,2:end])
[axx.xlabel= "x [μm]" for axx in ax_all[1:end-1,:]]
[axx.ylabel= "y [μm]" for axx in ax_all[:,1]]
[ axx.aspect=DataAspect() for axx in ax_all ]
linkaxes!(ax_all...)

txt= fig[5,2] = indicator(fig)
# for axx in ax_all
# 	on(mouseposition) do mpos
#
# end

fig

##

function _𝓟x!(Ho::AbstractArray{Complex{T},4},Hi::AbstractArray{Complex{T},4},grid::Grid{3}) where T<:Real
    Nx,Ny,Nz = size(grid)
	temp1 = zero(eltype(Ho))
	temp2 = zero(eltype(Ho))
    @avx for iz ∈ 1:Nz, iy ∈ 1:Ny, ix ∈ 1:((Nx÷2)-1), l in 0:0
		temp1 		= 0.5*( Hi[1+l,ix+1,iy,iz] - Hi[1+l,Nx-ix+1,iy,iz] )
		temp2 		= 0.5*( Hi[2+l,ix+1,iy,iz] + Hi[2+l,Nx-ix+1,iy,iz] )
		Ho[1+l,ix+1,iy,iz] 	= temp1
		Ho[2+l,ix+1,iy,iz] 	= temp2
		Ho[1+l,Nx-ix+1,iy,iz] = -temp1
		Ho[2+l,Nx-ix+1,iy,iz] = temp2
	end
	return Ho
end

function _𝓟x̄!(Ho::AbstractArray{Complex{T},4},Hi::AbstractArray{Complex{T},4},grid::Grid{3}) where T<:Real
    Nx,Ny,Nz = size(grid)
	temp1 = zero(eltype(Ho))
	temp2 = zero(eltype(Ho))
    @avx for iz ∈ 1:Nz, iy ∈ 1:Ny, ix ∈ 1:((Nx÷2)-1), l in 0:0
		temp1 		= 0.5*( Hi[1+l,ix+1,iy,iz] + Hi[1+l,Nx-ix+1,iy,iz] )
		temp2 		= 0.5*( Hi[2+l,ix+1,iy,iz] - Hi[2+l,Nx-ix+1,iy,iz] )
		Ho[1+l,ix+1,iy,iz] 	= temp1
		Ho[2+l,ix+1,iy,iz] 	= temp2
		Ho[1+l,Nx-ix+1,iy,iz] = temp1
		Ho[2+l,Nx-ix+1,iy,iz] = -temp2
	end
	return Ho
end


##
Nx = 128
Ny = 128


for ix ∈ 0:(Nx÷2) #1:((Nx÷2)-1), l in 0:0
	ij = ix #(iy-1) * Ny + ix-1
    ij2 = (ix > 0 ? Nx - ix : 0) # (iy-1) * Ny + ((ix-1) > 0 ? Nx - (ix-1) : 0)
	println("")
	println("ij, ij2: $ij , $ij2")
	println("")
end

for ix ∈ 1:(Nx÷2)+1 #1:((Nx÷2)-1), l in 0:0
	# ix1 = ix + 1
	# ix2 = (ix > 0 ? Nx - ix : 0) + 1
	ix1 = ix
	ix2 = (ix > 1 ? Nx - (ix-2) : 1)
	println("")
	println("ix1, ix2: $ix1 , $ix2")
	println("")
end



##
ms = ModeSolver(1.45, rwg(p), grid, nev=4)
ω = 0.75
k,H⃗ = solve_k(ω,rwg(p),grid)
ms.H⃗ |> size
##
H1 = copy(ms.H⃗[:,1])
H2 = copy(ms.H⃗[:,2])
H3 = copy(ms.H⃗[:,3])
H4 = copy(ms.H⃗[:,4])
Px! = 𝓟x!(ms.grid)
Px̄! = 𝓟x̄!(ms.grid)

H1c = copy(H1)
Px!*H1c

H1c = copy(H1)
mul!(H1c,Px!,H1c)

H1c = copy(H1)
maximum(abs2.(mul!(H1c,Px!,H1c) .- H1))

H1c = copy(H1)
maximum(abs2.(mul!(H1c,Px̄!,H1c) .- H1))

H1c = copy(H1)
maximum(abs2.(Px!*H1c .- H1))

H1c = copy(H1)
maximum(abs2.(Px̄!*H1c .- H1))


H1c = copy(H1)
maximum(abs2.(Px!2(Hrc,gr,-1) .- Hr))
using LinearAlgebra
mul!(H1c,Px!,H1c)

vec(_𝓟x!(reshape(H1c,(2,grid.Nx,grid.Ny)),grid))
33
33
33

##
# function _𝓟y!(H::AbstractArray{Complex{T},3},grid::Grid{2},parity=1) where T<:Real
#     Nx,Ny = size(grid)
# 	temp1 = zero(eltype(H))
# 	temp2 = zero(eltype(H))
#     @avx for iy ∈ 1:((Ny÷2)-1), ix ∈ 1:Nx, l in 0:0
# 		temp1 		= 0.5*( H[1+l,ix,iy+1] - parity * H[1+l,ix,Ny-iy+1] )
# 		temp2 		= 0.5*( H[2+l,ix,iy+1] + parity * H[2+l,ix,Ny-iy+1] )
# 		H[1+l,ix,iy+1] 	= temp1
# 		H[2+l,ix,iy+1] 	= temp2
# 		H[1+l,ix,Ny-iy+1] = -parity 	* 	temp1
# 		H[2+l,ix,Ny-iy+1] = parity 		* 	temp2
# 	end
# 	return H
# end
#
# function _𝓟y!(H::AbstractArray{Complex{T},4},grid::Grid{3},parity=1) where T<:Real
#     Nx,Ny,Nz = size(grid)
# 	temp1 = zero(eltype(H))
# 	temp2 = zero(eltype(H))
#     @avx for iz ∈ 1:Nz, iy ∈ 1:((Ny÷2)-1), ix ∈ 1:Nx, l in 0:0
# 		temp1 		= 0.5*( H[1+l,ix,iy+1,iz] - parity * H[1+l,ix,Ny-iy+1,iz] )
# 		temp2 		= 0.5*( H[2+l,ix,iy+1,iz] + parity * H[2+l,ix,Ny-iy+1,iz] )
# 		H[1+l,ix,iy+1,iz] 	= temp1
# 		H[2+l,ix,iy+1,iz] 	= temp2
# 		H[1+l,ix,Ny-iy+1,iz] = -parity 	* 	temp1
# 		H[2+l,ix,Ny-iy+1,iz] = parity 		* 	temp2
# 	end
# 	return H
# end
#
# function _𝓟ₓ!(H::StructArray{Complex{T},3},grid::Grid{2},parity=1) where T<:Real
# 	𝓟ₓ!(H.re,grid,parity)
# 	𝓟ₓ!(H.im,grid,parity)
# 	return nothing
# end

## test code
#
# Hrc = copy(Hr)
# maximum(abs2.(Px!1(Hrc,gr,-1) .- Hr))
# Hrc = copy(Hr)
# maximum(abs2.(Px!2(Hrc,gr,-1) .- Hr))
#
# Hr2c = copy(Hr2)
# maximum(abs2.(Px!1(Hr2c,gr,1) .- Hr2))
# Hr2c = copy(Hr2)
# maximum(abs2.(𝓟ₓ!3(Hr2c,gr,1) .- Hr2))
# # maximum(abs2.(Px!2(Hr2c,gr,1) .- Hr2))
#
# Hrstc = copy(Hrst)
# maximum(abs2.(Px!1(Hrstc,gr,-1) .- Hrst))
# Hrstc = copy(Hrst)
# maximum(abs2.(𝓟ₓ!(Hrstc,gr,-1) .- Hrst))
#
# @btime $𝓟ₓ!3($Hr2c,$gr,1)
# @btime 𝓟ₓ!($Hrstc,$gr,-1)
#
# 3
# 3
#
#  # test
# ##
##
using GLMakie
using GLMakie: lines!
H⃗ = OptiMode.H⃗
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,LNx,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       0.5,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];
geom = rwg(p)
# ms = ModeSolver(1.45, geom, grid, nev=4,constraint=𝓟x!(grid))
ms = ModeSolver(1.45, geom, grid, nev=4,) # constraint=𝓟x!(grid))
ω = 0.75
k,Hv = solve_k(ms,ω,rwg(p),nev=4,eigind=1)
k_old = 1.5167250170625406
k_new = k
k1 = copy(k)
abs(k_old - k_new) / k_new
HrN = Node(copy(reshape(ms.H⃗[:,1],(2,Ns...))))
##
using ArrayInterface
using RecursiveArrayTools
gr = ms.grid
shape = size(grid)
m⃗ = ms.M̂.m⃗ |> vec |> VectorOfArray
n⃗ = ms.M̂.n⃗ |> vec |> VectorOfArray
gv = g⃗(gr) |> vec |> VectorOfArray
xv = x⃗(gr) |> vec |> VectorOfArray
(xx,yy,zz),(gx,gy,gz),(nx,ny,nz),(mx,my,mz) = vecarr_to_vectors.((xv,gv,n⃗,m⃗))
xxr,yyr,zzr,gxr,gyr,gzr,nxr,nyr,nzr,mxr,myr,mzr = reshape.((xx,yy,zz,gx,gy,gz,nx,ny,nz,mx,my,mz),(shape,))

# length of eigenvector (H-field in recip space/transverse pol. basis is always N(grid)*2 = (N(grid)=Nx×Ny×Nz recip. lattice grid points) × (2 transverse polarizations for each plane wave)
N(gr)*2 == size(HH,1)
# x-mirror symmetry of flattened (by `vec`) x-value array (at each xyz point)
# @assert xx[end:-1:begin] == -xx
# x-mirror symmetry of flattened (by `vec`) y-value array (at each xyz point)
# @assert yy[end:-1:begin] == -yy
# x-mirror symmetry of flattened (by `vec`) gx-value array (at each xyz point)
# @assert gx[end:-1:begin] ≈ (-gx .- (1.0/gr.Δx))
# lines!(axes_g[1],-gx[end:-1:begin],color=:green)

# @assert mx[1:Nx÷2][2:end] == -mx[Nx:-1:Nx÷2+1][1:end-1]
# @assert mxs[1:Nx÷2][2:end] == -mxs[Nx:-1:Nx÷2+1][1:end-1]
#
# @assert gxr[1:Nx÷2,:][2:end,:] == -gxr[Nx:-1:Nx÷2+1,:][1:end-1,:]
# @assert mxr[1:Nx÷2,:][2:end,:] == -mxr[Nx:-1:Nx÷2+1,:][1:end-1,:]
#
# @assert gyr[:,1:Ny÷2][:,2:end] == -gyr[:,Ny:-1:Ny÷2+1][:,1:end-1]
# @assert myr[:,1:Ny÷2][:,2:end] == -myr[:,Ny:-1:Ny÷2+1][:,1:end-1]

# odd x-parity example
maximum(abs2.(Hr))
maximum(abs2.(Hr[1,1:Nx÷2,:][2:end,:] .- Hr[1,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are zero everywhere for odd parity
maximum(abs2.(Hr[1,1:Nx÷2,:][2:end,:] .+ Hr[1,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are nonzero everywhere for odd parity
maximum(abs2.(Hr[2,1:Nx÷2,:][2:end,:] .- Hr[2,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are nonzero everywhere for odd parity
maximum(abs2.(Hr[2,1:Nx÷2,:][2:end,:] .+ Hr[2,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are zero everywhere for odd parity

# direct indexing
Hr[1,2:Nx÷2,:] == Hr[1,1:Nx÷2,:][2:end,:]
Hr[1,Nx:-1:Nx÷2+2,:] == Hr[1,Nx:-1:Nx÷2+1,:][1:end-1,:]

# odd parity with direct indexing
parity = -1	# -1 => odd, 1 => even
Hrsym = similar(Hr)
Hrsym[1,2:Nx÷2,:] .= 0.5*( Hr[1,2:Nx÷2,:] - parity * Hr[1,Nx:-1:Nx÷2+2,:] )
Hrsym[2,2:Nx÷2,:] .= 0.5*( Hr[2,2:Nx÷2,:] + parity * Hr[2,Nx:-1:Nx÷2+2,:] )
Hrsym[1,Nx:-1:Nx÷2+2,:] .= -parity * Hrsym[1,2:Nx÷2,:]
Hrsym[2,Nx:-1:Nx÷2+2,:] .=  parity * Hrsym[2,2:Nx÷2,:]

maximum(abs2.(Hrsym))
maximum(abs2.(Hrsym[1,1:Nx÷2,:][2:end,:] .- Hrsym[1,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are zero everywhere for odd parity
maximum(abs2.(Hrsym[1,1:Nx÷2,:][2:end,:] .+ Hrsym[1,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are nonzero everywhere for odd parity
maximum(abs2.(Hrsym[2,1:Nx÷2,:][2:end,:] .- Hrsym[2,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are nonzero everywhere for odd parity
maximum(abs2.(Hrsym[2,1:Nx÷2,:][2:end,:] .+ Hrsym[2,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are zero everywhere for odd parity
# these are zero everywhere for when the parity operation left Hrsym ≈ Hr:
maximum(abs2.(Hrsym[1,2:Nx÷2,:] .- Hr[1,2:Nx÷2,:]))
maximum(abs2.(Hrsym[1,Nx:-1:Nx÷2+2,:] .- Hr[1,Nx:-1:Nx÷2+2,:]))
maximum(abs2.(Hrsym[2,2:Nx÷2,:] .- Hr[2,2:Nx÷2,:]))
maximum(abs2.(Hrsym[2,Nx:-1:Nx÷2+2,:] .- Hr[2,Nx:-1:Nx÷2+2,:]))


# @assert Hr[1,1:Nx÷2,:][2:end,:] == -Hr[1,Nx:-1:Nx÷2+1,:][1:end-1,:]
# @assert Hr[2,:,1:Ny÷2][:,2:end] == Hr[2,:,Ny:-1:Ny÷2+1][:,1:end-1]

# even x-parity example
Hr2 = reshape(HH[:,1],(2,Nx,Ny))
maximum(abs2.(Hr2))
maximum(abs2.(Hr2[1,1:Nx÷2,:][2:end,:] .- Hr2[1,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are nonzero everywhere for even parity
maximum(abs2.(Hr2[1,1:Nx÷2,:][2:end,:] .+ Hr2[1,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are zero everywhere for even parity
maximum(abs2.(Hr2[2,1:Nx÷2,:][2:end,:] .- Hr2[2,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are zero everywhere for even parity
maximum(abs2.(Hr2[2,1:Nx÷2,:][2:end,:] .+ Hr2[2,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are nonzero everywhere for even parity

# even parity with direct indexing
parity = 1	# -1 => odd, 1 => even
Hr2sym = similar(Hr2)
Hr2sym[1,2:Nx÷2,:] .= 0.5*( Hr2[1,2:Nx÷2,:] - parity * Hr2[1,Nx:-1:Nx÷2+2,:] )
Hr2sym[2,2:Nx÷2,:] .= 0.5*( Hr2[2,2:Nx÷2,:] + parity * Hr2[2,Nx:-1:Nx÷2+2,:] )
Hr2sym[1,Nx:-1:Nx÷2+2,:] .= -parity * Hr2sym[1,2:Nx÷2,:]
Hr2sym[2,Nx:-1:Nx÷2+2,:] .=  parity * Hr2sym[2,2:Nx÷2,:]

maximum(abs2.(Hr2sym))
maximum(abs2.(Hr2sym[1,1:Nx÷2,:][2:end,:] .- Hr2sym[1,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are zero everywhere for odd parity
maximum(abs2.(Hr2sym[1,1:Nx÷2,:][2:end,:] .+ Hr2sym[1,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are nonzero everywhere for odd parity
maximum(abs2.(Hr2sym[2,1:Nx÷2,:][2:end,:] .- Hr2sym[2,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are nonzero everywhere for odd parity
maximum(abs2.(Hr2sym[2,1:Nx÷2,:][2:end,:] .+ Hr2sym[2,Nx:-1:Nx÷2+1,:][1:end-1,:]))	# <--- these are zero everywhere for odd parity
# these are zero everywhere for when the parity operation left Hr2sym ≈ Hr2:
maximum(abs2.(Hr2sym[1,2:Nx÷2,:] .- Hr2[1,2:Nx÷2,:]))
maximum(abs2.(Hr2sym[1,Nx:-1:Nx÷2+2,:] .- Hr2[1,Nx:-1:Nx÷2+2,:]))
maximum(abs2.(Hr2sym[2,2:Nx÷2,:] .- Hr2[2,2:Nx÷2,:]))
maximum(abs2.(Hr2sym[2,Nx:-1:Nx÷2+2,:] .- Hr2[2,Nx:-1:Nx÷2+2,:]))

##
gxs = fftshift(gxr)|>vec
mxs = fftshift(mxr)|>vec
nxs = fftshift(nxr)|>vec
gys = fftshift(gyr)|>vec
mys = fftshift(myr)|>vec
nys = fftshift(nyr)|>vec
lines!(axes_g[1],gxs,color=:black)
lines!(axes_g[2],gys,color=:black)
lines!(axes_m[1],mxs,color=:black)
lines!(axes_m[2],mys,color=:black)
lines!(axes_n[1],nxs,color=:black)
lines!(axes_n[2],nys,color=:black)
##
# x-mirror symmetry of flattened (by `vec`) gy-value array (at each xyz point)
@assert gy[end:-1:begin] ≈ (-gy .- (1.0/gr.Δy))


##
fig = Figure()
axes_xy = fig[1:2,1] = [Axis(fig) for i=1:2]
axes_g = fig[1:2,2] = [Axis(fig) for i=1:2]
axes_m = fig[1,3:5] = [Axis(fig) for i=1:3]
axes_n = fig[2,3:5] = [Axis(fig) for i=1:3]
lxys = map((axx,xy,cc,ll)->(lines!(axx,xy,color=cc,label=ll);axislegend(axx)), axes_xy, [xx,yy], [:purple,:black],["x","y"])
lgs = map((axx,gg,cc,ll)->(lines!(axx,gg,color=cc,label=ll);axislegend(axx)), axes_g, [gx,gy], [:magenta,:cyan],["gx","gy"])
lns = map((axx,nn,cc,ll)->(lines!(axx,nn,color=cc,label=ll);axislegend(axx)), axes_n, [nx,ny,nz], [:red,:blue,:green],["nx","ny","nz"])
lms = map((axx,mm,cc,ll)->(lines!(axx,mm,color=cc,label=ll);axislegend(axx)), axes_m, [mx,my,mz], [:red,:blue,:green],["mx","my","mz"])
linkxaxes!(vcat(axes_xy,axes_g,axes_m,axes_n)...)
fig
##

##
HH = copy(ms.H⃗) # recip. space H-field soln. copy
Ns = size(ms.grid)
eigind=1
Hr = reshape(HH[:,eigind],(2,Ns...))
geom = rwg(p)
HrN[] = Hr; #reshape(HH[:,eigind],(2,Ns...));
E0 = E⃗(k,to_value(HrN),ω,geom,grid; svecs=false, normalized=false);
H0 = H⃗(k,to_value(HrN),ω,geom,grid; svecs=false, normalized=false);

E⃗1	= Node(E0)
Ex1	= Node(E0[1,:,:])
Ey1	= Node(E0[1,:,:])
E⃗Pp	= Node(E0)
ExPp	= Node(E0[1,:,:])
EyPp	= Node(E0[1,:,:])
E⃗Pn	= Node(E0)
ExPn	= Node(E0[1,:,:])
EyPn	= Node(E0[1,:,:])

H⃗1	= Node(H0)
Hx1	= Node(H0[1,:,:])
Hy1	= Node(H0[1,:,:])
H⃗Pp	= Node(H0)
HxPp	= Node(H0[1,:,:])
HyPp	= Node(H0[1,:,:])
H⃗Pn	= Node(H0)
HxPn	= Node(H0[1,:,:])
HyPn	= Node(H0[1,:,:])

##
HPp = vec(copy(Hr))
mul!(HPp,𝓟x!(ms.grid),vec(copy(Hr)));
# mul!(HPp,Pxs,vec(copy(Hr)));
HrPp = reshape(HPp,(2,Ns...));

HPn = vec(copy(Hr))
mul!(HPn,𝓟x̄!(ms.grid),vec(copy(Hr)));
# mul!(HPn,Pxbs,vec(copy(Hr)));
HrPn = reshape(HPn,(2,Ns...));


HrN = Node(Hr)

H⃗1[] = H⃗(k1,Hr,ω,geom,grid; svecs=false, normalized=false)
Hx1[] = real(H⃗1[][1,:,:])
Hy1[] = real(H⃗1[][2,:,:])


H⃗Pp[] = H⃗(k1,HrPp,ω,geom,grid; svecs=false, normalized=false)
HxPp[] = real(H⃗Pp[][1,:,:])
HyPp[] = real(H⃗Pp[][2,:,:])

H⃗Pn[] = H⃗(k1,HrPn,ω,geom,grid; svecs=false, normalized=false)
HxPn[] = real(H⃗Pn[][1,:,:])
HyPn[] = real(H⃗Pn[][2,:,:])

E⃗1[] = E⃗(k1,Hr,ω,geom,grid; svecs=false, normalized=false)
Ex1[] = real(E⃗1[][1,:,:])
Ey1[] = real(E⃗1[][2,:,:])

E⃗Pp[] = E⃗(k1,HrPp,ω,geom,grid; svecs=false, normalized=false)
ExPp[] = real(E⃗Pp[][1,:,:])
EyPp[] = real(E⃗Pp[][2,:,:])

E⃗Pn[] = E⃗(k1,HrPn,ω,geom,grid; svecs=false, normalized=false)
ExPn[] = real(E⃗Pn[][1,:,:])
EyPn[] = real(E⃗Pn[][2,:,:])

# on(HrN) do HrN
#
# 	HrPp = copy(to_value(HrN))
# 	_𝓟x!(HrPp,copy(Hr),ms.grid);
# 	# HrPp = reshape(HPp,(2,Ns...)); similar(Hr);
# 	HrPn = copy(to_value(HrN))
# 	_𝓟x̄!(HrPn,copy(Hr),ms.grid);
# 	# HrPn = reshape(HPp,(2,Ns...)); similar(Hr);
#
# 	H⃗1[] = H⃗(k1,to_value(HrN),ω,geom,grid; svecs=false, normalized=false)
# 	Hx1[] = real(H⃗1[][1,:,:])
# 	Hy1[] = real(H⃗1[][2,:,:])
#
#
# 	H⃗Pp[] = H⃗(k1,HrPp,ω,geom,grid; svecs=false, normalized=false)
# 	HxPp[] = real(H⃗Pp[][1,:,:])
# 	HyPp[] = real(H⃗Pp[][2,:,:])
#
# 	H⃗Pn[] = H⃗(k1,HrPn,ω,geom,grid; svecs=false, normalized=false)
# 	HxPn[] = real(H⃗Pn[][1,:,:])
# 	HyPn[] = real(H⃗Pn[][2,:,:])
#
# 	E⃗1[] = E⃗(k1,Hr,ω,geom,grid; svecs=false, normalized=false)
# 	Ex1[] = real(E⃗1[][1,:,:])
# 	Ey1[] = real(E⃗1[][2,:,:])
#
# 	E⃗Pp[] = E⃗(k1,HrPp,ω,geom,grid; svecs=false, normalized=false)
# 	ExPp[] = real(E⃗Pp[][1,:,:])
# 	EyPp[] = real(E⃗Pp[][2,:,:])
#
# 	E⃗Pn[] = E⃗(k1,HrPn,ω,geom,grid; svecs=false, normalized=false)
# 	ExPn[] = real(E⃗Pn[][1,:,:])
# 	EyPn[] = real(E⃗Pn[][2,:,:])
#
# 	Hs .= reshape([  Hx1,  Hy1, HxPp ,HyPp ,  HxPn ,HyPn ],(2,3))
# 	Es .= reshape([  Ex1,  Ey1, ExPp ,EyPp ,  ExPn ,EyPn ],(2,3))
# 	Hmagmax .= [lift(A->maximum(abs,A),X) for X in Hs]
# 	Emagmax .= [lift(A->maximum(abs,A),X) for X in Es]
# end
##
# using LoopVectorization
# Pxr = zeros(eltype(Hr),size(Hr))
# Px̄r = zeros(eltype(Hr),size(Hr))
# @avx for iy ∈ 1:Ny
# 	for ix ∈ 2:(Nx÷2), l in 0:0 #1:((Nx÷2)-1), l in 0:0
# 		ix1 = ix
# 		ix2 = (ix > 1 ? Nx - (ix-2) : 1)
# 		Pxr[1+l,ix1,iy] = 1.0
# 		Pxr[2+l,ix1,iy] = 1.0
# 		Pxr[1+l,ix2,iy] = -1.0
# 		Pxr[2+l,ix2,iy] = 1.0
# 	end
# end
#
# @avx for iy ∈ 1:Ny
# 	for ix ∈ 2:(Nx÷2), l in 0:0 #1:((Nx÷2)-1), l in 0:0
# 		ix1 = ix
# 		ix2 = (ix > 1 ? Nx - (ix-2) : 1)
# 		Px̄r[1+l,ix1,iy] = 1.0
# 		Px̄r[2+l,ix1,iy] = 1.0
# 		Px̄r[1+l,ix2,iy] = 1.0
# 		Px̄r[2+l,ix2,iy] = -1.0
# 	end
# end
#
# Pxv = vec(Pxr) #.* inv(N(grid))
# Px̄v = vec(Px̄r) #.* inv(N(grid))


##
#
# H⃗Px = H⃗(k1,Pxr,ω,geom,grid; svecs=false, normalized=false)
# HPxxr =  sign.(real(H⃗Px[1,:,:]) )
# HPxyr =  sign.(real(H⃗Px[2,:,:]) )
# HPxzr =  sign.(real(H⃗Px[3,:,:]) )
# HPxxi = sign.( imag(H⃗Px[1,:,:]) )
# HPxyi =  sign.(imag(H⃗Px[2,:,:]) )
# HPxzi =  sign.(imag(H⃗Px[3,:,:]) )
#
# fig = Figure()
#
# Hs = reshape([  HPxxr, HPxxi, HPxyr, HPxyi, HPxzr, HPxzi,    ],(2,3))
# Hmagmax = [maximum(abs,A) for A in Hs]
# cmaps_H = [:diverging_bwr_40_95_c42_n256, :diverging_protanopic_deuteranopic_bwy_60_95_c32_n256]
# pos_H = [fig[i,j] for i=1:2,j=1:3]
# ax_H = [Axis(pos[1,1]) for pos in pos_H]
# label_base = reshape( ["xr","xi", "yr", "yi", "zr", "zi",], (2,3))
# labels_H = "H".*label_base
# heatmaps_H = [heatmap!(ax_H[i,j], xs, ys, Hs[i,j],colormap=cmaps_H[i],label=labels_H[i,j],colorrange=(-Hmagmax[i,j],Hmagmax[i,j])) for i=1:2,j=1:3]
# cbars_H = [Colorbar(pos_H[i,j][1, 2], heatmaps_H[i,j],  width=20 ) for i=1:2,j=1:3]
# map( (axx,ll)->text!(axx,ll,position=(-1.4,1.1),textsize=0.7,color=:black), ax_H, labels_H )
# ax_all = vcat(ax_H)
# hidexdecorations!.(ax_all[1:end-1,:])
# hideydecorations!.(ax_all[:,2:end])
# [axx.xlabel= "x [μm]" for axx in ax_all[1:end-1,:]]
# [axx.ylabel= "y [μm]" for axx in ax_all[:,1]]
# [ axx.aspect=DataAspect() for axx in ax_all ]
# linkaxes!(ax_all...)
#
# txt= fig[5,2] = indicator(fig)
# # for axx in ax_all
# # 	on(mouseposition) do mpos
# #
# # end
#
# fig

## interactions

function indicator(ax::Axis,ob)
    register_interaction!(ax, :indicator) do event::MouseEvent, axis
    if event.type === MouseEventTypes.over
        ob[] = event.data
    end
    end
end
function indicator(grid::GridLayout,ob)
    foreach(Axis,grid;recursive=true) do ax
    indicator(ax,ob)
    end
end
function indicator(grid::GridLayout)
    ob = Observable(Point2f0(0.,0.))
    indicator(grid,ob)
    ob
end
function indicator(fig,args...; tellwidth=false, kwargs...)
    Label(
        fig,
        lift(ind->"x: $(ind[1])  y: $(ind[2])",indicator(fig.layout)),
        args...; tellwidth=tellwidth, kwargs...
    )
end




## H and E fields
using AbstractPlotting: mouseposition
using AbstractPlotting

Hs = reshape([  Hx1,  Hy1, HxPp ,HyPp ,  HxPn ,HyPn ],(2,3))
Es = reshape([  Ex1,  Ey1, ExPp ,EyPp ,  ExPn ,EyPn ],(2,3))
Hmagmax = [lift(A->maximum(abs,A),X) for X in Hs]
Emagmax = [lift(A->maximum(abs,A),X) for X in Es]

fig = Figure()
xs = x(ms.grid)
ys = y(ms.grid)

pos_H = [fig[i,j] for i=1:2,j=1:3]
ax_H = [Axis(pos[1,1]) for pos in pos_H]
pos_E = [fig[i,j] for i=3:4,j=1:3]
ax_E = [Axis(pos[1,1]) for pos in pos_E]

# nx = sqrt.(getindex.(inv.(ms.M̂.ε⁻¹),1,1))
cmaps_H = [:diverging_bwr_40_95_c42_n256, :diverging_protanopic_deuteranopic_bwy_60_95_c32_n256]
cmaps_E = [:diverging_bkr_55_10_c35_n256, :diverging_bky_60_10_c30_n256]
label_base = reshape( ["x","y","xPp","yPp","xPn","yPn",], (2,3))
labels_H = "H".*label_base
labels_E = "E".*label_base
heatmaps_H = [heatmap!(ax_H[i,j], xs, ys, Hs[i,j],colormap=cmaps_H[i],label=labels_H[i,j],colorrange=(-to_value(Hmagmax[i,j]),to_value(Hmagmax[i,j]))) for i=1:2,j=1:3]
heatmaps_E = [heatmap!(ax_E[i,j], xs, ys, Es[i,j],colormap=cmaps_E[i],label=labels_E[i,j],colorrange=(-to_value(Emagmax[i,j]),to_value(Emagmax[i,j]))) for i=1:2,j=1:3]
cbars_H = [Colorbar(pos_H[i,j][1, 2], heatmaps_H[i,j],  width=20 ) for i=1:2,j=1:3]
cbars_E = [Colorbar(pos_E[i,j][1, 2], heatmaps_E[i,j],  width=20 ) for i=1:2,j=1:3]

map( (axx,ll)->text!(axx,ll,position=(-1.4,1.1),textsize=0.7,color=:black), ax_H, labels_H )
map( (axx,ll)->text!(axx,ll,position=(-1.4,1.1),textsize=0.7,color=:white), ax_E, labels_E )
ax_all = vcat(ax_H,ax_E)
hidexdecorations!.(ax_all[1:end-1,:])
hideydecorations!.(ax_all[:,2:end])
[axx.xlabel= "x [μm]" for axx in ax_all[1:end-1,:]]
[axx.ylabel= "y [μm]" for axx in ax_all[:,1]]
[ axx.aspect=DataAspect() for axx in ax_all ]
linkaxes!(ax_all...)

txt= fig[5,2] = indicator(fig)
# for axx in ax_all
# 	on(mouseposition) do mpos
#
# end

fig

## E fields only

# Es = reshape([  Ex1,  Ey1, ExPp ,EyPp ,  ExPn ,EyPn ],(2,3))
# fig = Figure()
#
#
# xs = x(ms.grid)
# ys = y(ms.grid)
#
# ax_nx = fig[1,2] = Axis(fig)
# ax_E = fig[2:3,1:3] = reshape( [ Axis(fig) for i=1:6 ], (2,3)) #, title = t) for t in [ "|Eₓ|² @ ω", "|Eₓ|² @ 2ω" ] ]
#
# nx = sqrt.(getindex.(inv.(ms.M̂.ε⁻¹),1,1))
#
# # cmaps_E = [:linear_ternary_red_0_50_c52_n256, :linear_ternary_blue_0_44_c57_n256]
# cmaps_E = [:berlin, :lisbon]
# labels_E = ["rel. |Eₓ|² @ ω","rel. |Ey|² @ ω"]
# heatmaps_E = [heatmap!(ax_E[i,j], xs, ys, Es[i,j],colormap=cmaps_E[i],label=labels_E[i]) for i=1:2,j=1:3]
# hm_nx = heatmap!(ax_nx,xs,ys,nx;colormap=:viridis)
# text!(ax_nx,"nₓ",position=(1.4,1.1),textsize=0.7,color=:white)
# text!(ax_E[1,1],"real Eₓ (ω)",position=(-1.4,1.1),textsize=0.7,color=:white)
# text!(ax_E[2,1],"real Ey (ω)",position=(-1.7,1.1),textsize=0.7,color=:white)
# text!(ax_E[1,2],"real EₓPp (ω)",position=(-1.4,1.1),textsize=0.7,color=:white)
# text!(ax_E[2,2],"real EyPp (ω)",position=(-1.7,1.1),textsize=0.7,color=:white)
# text!(ax_E[1,3],"real EₓPn (ω)",position=(-1.4,1.1),textsize=0.7,color=:white)
# text!(ax_E[2,3],"real EyPn (ω)",position=(-1.7,1.1),textsize=0.7,color=:white)
# cbar_nx = fig[1,4] = Colorbar(fig,hm_nx ) #,label="nₓ")
# cbar_Ex = fig[2,4] = Colorbar(fig,heatmaps_E[1]) #,label="rel. |Eₓ|² @ ω")
# cbar_Ey = fig[3,4] = Colorbar(fig,heatmaps_E[2]) #,label="rel. |Eₓ|² @ 2ω")
# for cb in [cbar_Ex, cbar_Ey, cbar_nx]
#     cb.width=30
#     cb.height=Relative(2/3)
# end
# # label, format
# # hidexdecorations!(ax_E[1])
# hidexdecorations!(ax_nx)
#
# ax_E[1].ylabel = "y [μm]"
# ax_E[2].ylabel = "y [μm]"
# ax_E[2].xlabel = "x [μm]"
#
# ax_nx.aspect=DataAspect()
# all_axes = vcat(ax_nx,ax_E)
# [ aa.aspect=DataAspect() for aa in all_axes ]
# linkxaxes!(all_axes...)
# fig
