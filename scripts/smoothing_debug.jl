using Revise
using OptiMode
using LinearAlgebra, Statistics, ArrayInterface, RecursiveArrayTools, StaticArrays, HybridArrays
using GeometryPrimitives, BenchmarkTools
using ChainRules, Zygote, ForwardDiff, FiniteDifferences
using UnicodePlots
using Crayons.Box       # for color printing
using Zygote: @ignore, dropgrad
p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
       # 0.5,                #   vacuum gap at boundaries `edge_gap`     [μm]
               ];
Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 128, 128, 1;
# Δx,Δy,Δz,Nx,Ny,Nz = 6.0, 4.0, 1.0, 256, 256, 1;
gr = Grid(Δx,Δy,Nx,Ny)
# rwg(x) = ridge_wg(x[1],x[2],x[3],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # dispersive material model version
rwg(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
rwg2(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,2.2,1.4,Δx,Δy) # constant index version

p = [
       1.7,                #   top ridge width         `w_top`         [μm]
       0.7,                #   ridge thickness         `t_core`        [μm]
       0.5,                #   ridge thickness         `t_core`        [μm]
       π / 14.0,           #   ridge sidewall angle    `θ`             [radian]
               ];
geom = rwg(p)
ms = ModeSolver(1.45, geom, gr)
ωs = [0.65, 0.75]

##
# using StaticArrays: Dynamic
using IterativeSolvers: bicgstabl
using LinearAlgebra, FFTW
function ∂²ω²∂k²_manual(x)
	ms = ModeSolver(1.45, geom, gr)
	k,H⃗ = solve_k(ms,x,geom)
	ε⁻¹ = deepcopy(ms.M̂.ε⁻¹)
	# mag,m⃗,n⃗ = mag_m_n(k,M̂.g⃗)
	# mag = copy(ms.M̂.mag
	# m = copy(ms.M̂.m # HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,m⃗))
	# n = copy(ms.M̂.n # HybridArray{Tuple{3,Dynamic(),Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,n⃗))
	k̄ = ∂²ω²∂k²(x^2,copy(H⃗),copy(k),ε⁻¹,gr)
end

function foo1(H⃗::AbstractVector{Complex{T}},ε⁻¹,mag,m,n) where T<:Real
	H = reshape(H⃗,(2,size(mag)...))
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
	X = zx_tc(H,mn) + kx_tc(H,mn,mag)
	Y = ifft( ε⁻¹_dot( fft( X, (2:3) ), real(flat(ε⁻¹))), (2:3))
	# -(kx_ct(Y,mn,mag) + zx_ct(Y,mn))
	dot(X,Y)
end

function Mₖᵀ_plus_Mₖ2(H⃗::AbstractVector{Complex{T}},ε⁻¹,mag,m,n) where T<:Real
	H = reshape(H⃗,(2,size(mag)...))
	mn = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
	X = zx_tc(H,mn) + kx_tc(H,mn,mag)
	Y = ifft( ε⁻¹_dot( fft( X, (2:3) ), real(flat(ε⁻¹))), (2:3))
	-(kx_ct(Y,mn,mag) + zx_ct(Y,mn))
end

function ∂²ω²∂k²2(ω²,H⃗,k,ε⁻¹,grid::Grid{ND,T};eigind=1,log=true) where {ND,T<:Real}
	M̂ = HelmholtzMap(k,ε⁻¹,grid)
	Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
	Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
	H = reshape(H⃗[:,eigind],(2,Ns...))
	g⃗s = g⃗(dropgrad(grid))
	(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
	m = M̂.m
	n = M̂.n
	ω = sqrt(ω²)
	HMₖH, HMₖH_pb = Zygote.pullback(H_Mₖ_H,H,ε⁻¹,mag,m,n)
	H̄2, eī2, māg2,m̄2,n̄2 = HMₖH_pb(1)
	m̄v2 = copy(reinterpret(reshape,SVector{3,T},real(m̄2)))
	n̄v2 = copy(reinterpret(reshape,SVector{3,T},real(n̄2)))
	∂ω²∂k = 2 * real(HMₖH[eigind])
	# println("typeof(māg2): $(typeof(māg2))")
	# println("typeof(m̄2): $(typeof(m̄2))")
	# println("typeof(n̄2): $(typeof(n̄2))")
	# println("size(māg2): $(size(māg2))")
	# println("size(m̄2): $(size(m̄2))")
	# println("size(n̄2): $(size(n̄2))")
	k̄₁ = mag_m_n_pb( (real(māg2), m̄v2, n̄v2) )[1]
	# k̄₁ = -mag_m_n_pb(( māg2, m̄2, n̄2 ))[1]	# should equal ∂/∂k(2 * ∂ω²/∂k) = 2∂²ω²/∂k²


	# mn = vcat(reshape(M̂.m,(1,size(M̂.m)...)),reshape(M̂.n,(1,size(M̂.m)...)))
	# H̄ = vec(Mₖᵀ_plus_Mₖ(H⃗[:,eigind],ε⁻¹,mag,m,n))
	Mkop = M̂ₖ_sp(ω,k,geom,grid)
	H̄ = (Mkop + transpose(Mkop)) * ms.H⃗[:,eigind]

	println("manual backsolve:")
	println("man. Hbar_magmax = $(maximum(abs2.(H̄)))")
	println("Hbar2_magmax = $(maximum(abs2.(H̄2)))")
	# println("size(H̄): $(size(H̄))")
	# println("size(H̄2): $(size(H̄2))")
	# H̄ = vec(H̄2)
	adj_res = ∂ω²∂k_adj(M̂,ω²,H⃗,H̄;eigind,log)
	if !log
		λ⃗₀ = adj_res
	else
		show(adj_res[2])
		println("")
		# show(uplot(adj_res[2]))
		# println("")
		λ⃗₀ = adj_res[1]
	end
	# λ⃗₀ = !log ? adj_res : ( uplot(adj_res[2]); adj_res[1])
	println("man. lm0_magmax = $(maximum(abs2.(λ⃗₀)))")
	λ⃗ = λ⃗₀ - dot(H⃗[:,eigind],λ⃗₀) * H⃗[:,eigind] #+ H⃗[:,eigind]
	println("man. lm_magmax = $(maximum(abs2.(λ⃗)))")
	H = reshape(H⃗[:,eigind],(2,Ns...))
	λ = reshape(λ⃗,(2,Ns...))
	# zxh = M̂.𝓕 * kx_tc(H,mn,mag)  * M̂.Ninv # zx_tc(H,mn)  * M̂.Ninv
	# λd =  M̂.𝓕 * kx_tc(λ,mn,mag)
	eī = similar(ε⁻¹)
	λd = similar(M̂.d)
	λẽ = similar(M̂.d)
	# ε⁻¹_bar!(eī, vec(zxh), vec(λd), Ns...)
	# #TODO replace iffts below with pre-planned ifft carried by M̂
	# λẽf = fft( ε⁻¹_dot( (λd * M̂.Ninv), real(flat(ε⁻¹))), (2:3))
	# ẽf = fft( ε⁻¹_dot( zxh, real(flat(ε⁻¹))), (2:3))
	# λẽ = reinterpret(reshape, SVector{3,Complex{T}}, λẽf )
	# ẽ = reinterpret(reshape, SVector{3,Complex{T}}, ẽf )
	# # scaling by mag or √mag may differ from normal case here, as one of the kx
	# # operators has been replaced by ẑx, so two of the four terms in the next two
	# # lines are a factor of mag smaller at each point in recip. space?
	# kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
	# kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
	# māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
	d = _H2d!(M̂.d, H * M̂.Ninv, M̂) # =  M̂.𝓕 * kx_tc( H , mn2, mag )  * M̂.Ninv
	λd = _H2d!(λd,λ,M̂) # M̂.𝓕 * kx_tc( reshape(λ⃗,(2,M̂.Nx,M̂.Ny,M̂.Nz)) , mn2, mag )
	ε⁻¹_bar!(eī, vec(M̂.d), vec(λd), Ns...)
	# eīₕ = copy(ε⁻¹_bar)
	# back-propagate gradients w.r.t. `(k⃗+g⃗)×` operator to k via (m⃗,n⃗) pol. basis and |k⃗+g⃗|
	λd *=  M̂.Ninv
	λẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(λẽ , λd  ,M̂ ) )
	ẽ = reinterpret(reshape, SVector{3,Complex{T}}, _d2ẽ!(M̂.e,M̂.d,M̂) )
	kx̄_m⃗ = real.( λẽ .* conj.(view(H,2,Nranges...)) .+ ẽ .* conj.(view(λ,2,Nranges...)) )
	kx̄_n⃗ =  -real.( λẽ .* conj.(view(H,1,Nranges...)) .+ ẽ .* conj.(view(λ,1,Nranges...)) )
	māg = dot.(n⃗, kx̄_n⃗) + dot.(m⃗, kx̄_m⃗)
	# almost there! need to replace this pullback with a Zygote compatible fn.
	k̄₂ = -mag_m_n_pb(( māg, kx̄_m⃗.*mag, kx̄_n⃗.*mag ))[1]	# should equal ∂/∂k(2 * ∂ω²/∂k) = 2∂²ω²/∂k²
	println("k̄₁ = $(k̄₁)")
	println("k̄₂ = $(k̄₂)")
	k̄ = k̄₁ + k̄₂
	ω̄  =  (2 * sqrt(ω²) * k̄) / ∂ω²∂k #2ω * k̄ₖ / ∂ω²∂k[eigind]
	println("k̄ = k̄₁ + k̄₂ = $(k̄)")
	println("ω̄ = $(ω̄ )")
	return ω̄
end



##

using SparseArrays, FFTW, LinearMaps
kxtcsp = kx_tc_sp(k,gr)
vec(kx_tc(H,mns,mag)) ≈ kxtcsp * H⃗
vec(kx_ct(tc(H,mns),mns,mag)) ≈ -kxtcsp' * vec(tc(H,mns))
@btime $kxtcsp * $H⃗ # 163.864 μs (2 allocations: 768.08 KiB)
@btime vec(kx_tc($H,$mns,$mag)) # 378.265 μs (6 allocations: 768.34 KiB)

zxtcsp = zx_tc_sp(k,gr)
vec(zx_tc(H,mns)) ≈ zxtcsp * H⃗
vec(zx_ct(tc(H,mns),mns)) ≈ zxtcsp' * vec(tc(H,mns))
@btime $zxtcsp * $H⃗ # 151.754 μs (2 allocations: 768.08 KiB)
@btime vec(zx_tc($H,$mns)) # 296.939 μs (6 allocations: 768.38 KiB)

zx_tc_sp(k,gr) == zx_ct_sp(k,gr)'
# vec(zx_tc(H,mns)) ≈ zx_tc_sp_coo(mag,mns) * H⃗

eisp = ε⁻¹_sp(0.75,rwg(p),gr)
vec(ε⁻¹_dot(tc(H,mns),flat(εₛ⁻¹(0.75,rwg(p);ms)))) ≈ eisp * vec(tc(H,mns))

Mop = M̂_sp(ω,k,rwg(p),grid)
ms.M̂ * H⃗[:,eigind] ≈ Mop * H⃗[:,eigind]
ms.M̂ * ms.H⃗[:,eigind] ≈ Mop * ms.H⃗[:,eigind]
@btime $Mop * $H⃗[:,eigind] # 1.225 ms (122 allocations: 4.01 MiB)
@btime $ms.M̂ * $H⃗[:,eigind] # 4.734 ms (1535 allocations: 1.22 MiB)

Mkop = M̂ₖ_sp(ω,k,rwg(p),gr)
Mkop * H⃗[:,eigind] ≈ vec(-kx_ct( ifft( ε⁻¹_dot( fft( zx_tc(H,mns), (2:3) ), real(flat(ε⁻¹))), (2:3)),mns,mag))
@btime $Mkop * $H⃗[:,eigind] # 1.261 ms (122 allocations: 4.01 MiB)
@btime vec(-kx_ct( ifft( ε⁻¹_dot( fft( zx_tc($H,$mns), (2:3) ), real(flat(ε⁻¹))), (2:3)),$mns,$mag)) # 2.095 ms (94 allocations: 4.01 MiB)


nnginv = nngₛ⁻¹(ω,rwg(p),gr)
real(dot(H⃗[:,eigind],Mkop,H⃗[:,eigind])) ≈ H_Mₖ_H(H,ε⁻¹,mag,m,n)
real(dot(H⃗[:,eigind],Mkop,H⃗[:,eigind])) ≈ H_Mₖ_H(H,nnginv,mag,m,n)
@btime real(dot($H⃗[:,eigind],$Mkop,$H⃗[:,eigind])) # 1.465 ms (134 allocations: 4.51 MiB)
@btime H_Mₖ_H($H,$ε⁻¹,$mag,$m,$n) # 3.697 ms (122 allocations: 4.76 MiB)
#
# Zygote.gradient((om,kk,pp,HH)->real(dot(HH,M̂ₖ_sp(om,kk,rwg(pp),gr),HH)),ω,k,p,H⃗[:,eigind])
# Zygote.gradient((om,kk,pp,HH)->real(dot(HH,M̂ₖ_sp(om,kk,rwg(pp),gr)*HH)),ω,k,p,H⃗[:,eigind])

# ⟨H|Mₖ|H⟩

real(dot(H⃗[:,eigind],M̂ₖ_sp(ω,k,rwg(p),gr)*H⃗[:,eigind]))

# Zygote.gradient((a,b)->sum(foo2(a,b)),mag,mns)
# Zygote.gradient((a,b)->sum(abs2.(foo2(a,b))),mag,mns)

##

ω = 0.75
eigind = 1
grid = ms.grid
k,H⃗ = solve_k(ms,ω,rwg(p))
ε⁻¹ = ms.M̂.ε⁻¹
Ns = size(grid) # (Nx,Ny,Nz) for 3D or (Nx,Ny) for 2D
Nranges = eachindex(grid) #(1:NN for NN in Ns) # 1:Nx, 1:Ny, 1:Nz for 3D, 1:Nx, 1:Ny for 2D
H = reshape(H⃗[:,eigind],(2,Ns...))
g⃗s = g⃗(dropgrad(grid))
(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
m = ms.M̂.m
n = ms.M̂.n
mns = vcat(reshape(m,(1,size(m)[1],size(m)[2],size(m)[3])),reshape(n,(1,size(m)[1],size(m)[2],size(m)[3])))
HMₖH, HMₖH_pb = Zygote.pullback(H_Mₖ_H,H,ε⁻¹,mag,m,n)
H̄2, eī2, māg2,m̄2,n̄2 = HMₖH_pb(1)
m̄v2 = copy(reinterpret(reshape,SVector{3,Float64},real(m̄2)))
n̄v2 = copy(reinterpret(reshape,SVector{3,Float64},real(n̄2)))
∂ω²∂k = 2 * real(HMₖH)
k̄₁ = mag_m_n_pb( (real(māg2), m̄v2, n̄v2) )[1]

##

X1 = kx_tc(H,mns,mag)
X2 = tc(kx_ct(tc(H,mns),mns,mag),mns)
X1 ≈ X2
X3 = zx_tc(H,mns)
X4 = tc(zx_ct(tc(H,mns),mns),mns)
X3 ≈ X4


X = zx_tc(H,mns) + kx_tc(H,mns,mag)
Y = ifft( ε⁻¹_dot( fft( X, (2:3) ), real(flat(ε⁻¹))), (2:3))
# -(kx_ct(Y,mn,mag) + zx_ct(Y,mn))
dot(X,Y)

H̄1 = foo1(H⃗[:,eigind],ε⁻¹,mag,m,n)

zxtcH = zx_tc(H,mns)
H⃗dag = H⃗[:,eigind]'
Hdag1 = reshape(vec(H⃗[:,eigind]'),(2,Ns...))
Hdag2 = reshape(H⃗[:,eigind]',(2,Ns...))

##
fig = GLMakie.Figure()
@show H̄2_magmax = sqrt(maximum(abs2.(H̄2)))
@show H̄1_magmax = sqrt(maximum(abs2.(H̄1)))
H̄2_rel = H̄2 / H̄2_magmax
H̄1_rel = H̄1 / H̄1_magmax

axes_pb = fig[1,1:2] = [Axis(fig,title=t) for t in "|H̄_pb".*["1","2"].*"|²" ]
hms_pb = [GLMakie.heatmap!(axes_pb[axind],abs2.(fftshift(H̄2_rel[axind,:,:]))';colorrange=(0,1)) for axind=1:2]
cbar_pb = fig[1,3] = Colorbar(fig,hms_pb[1],label="relative mag. [1]")
cbar_pb.width = 30
axes_foo = fig[2,1:2] = [Axis(fig,title=t) for t in "|H̄_foo".*["1","2"].*"|²" ]
hms_foo = [GLMakie.heatmap!(axes_foo[axind],abs2.(fftshift(H̄1_rel[axind,:,:]))';colorrange=(0,1)) for axind=1:2]
cbar_foo = fig[2,3] = Colorbar(fig,hms_foo[1],label="relative mag. [1]")
cbar_foo.width = 30

axes = vcat(axes_pb,axes_foo) #,axes_Hi)
linkaxes!(axes...)
fig
##
om0 = 0.75
AD_style = BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style = BOLD*RED_FG
MAN_style = BOLD*GREEN_FG

AD_style_N = NEGATIVE*BOLD*BLUE_FG #NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
FD_style_N = NEGATIVE*BOLD*RED_FG
MAN_style_N = NEGATIVE*BOLD*GREEN_FG

# println(err_style("∂n_om_err:"))
##
println("")
println(AD_style_N("∂²ω²∂k²_AD:"))
println("")
∂²ω²∂k²_AD = Zygote.gradient(om->(om / solve_n(om,rwg(p),gr)[2]),om0)[1]
println(AD_style("∂²ω²∂k²_AD= $∂²ω²∂k²_AD"))
println("")

println("")
println(FD_style_N("∂²ω²∂k²_FD:"))
println("")
∂²ω²∂k²_FD = FiniteDifferences.central_fdm(5,1)(om->(om / solve_n(om,rwg(p),gr)[2]),om0)
println(FD_style("∂²ω²∂k²_FD: $∂²ω²∂k²_FD"))
println("")

println("")
println(MAN_style_N("∂²ω²∂k²_MAN:"))
println("")
∂²ω²∂k²_MAN = ∂²ω²∂k²(ω,rwg(p),k,H⃗,grid) #om0^2,H⃗,k,rwg(p),gr)
println(MAN_style("∂²ω²∂k²_MAN: $∂²ω²∂k²_MAN"))
println("")

##

∂ω_²ω²∂k²_RAD1, ∂p_²ω²∂k²_RAD1 = Zygote.gradient((om,x)->∂²ω²∂k²(om,εₛ⁻¹(om,rwg(x),grid),nngₛ⁻¹(om,rwg(x),grid),k,H⃗,grid)[1],
	ω,p)







∂ω_²ω²∂k²_RAD2, ∂p_²ω²∂k²_RAD2 = Zygote.gradient((om,x)->∂²ω²∂k²(real(om),rwg(real(x)),k,H⃗,grid),ω,p)





∂ω_²ω²∂k²_FD, ∂p_²ω²∂k²_FD = FiniteDifferences.grad(central_fdm(9,1),
		(om,x)->∂²ω²∂k²(om,εₛ⁻¹(om,rwg(x),grid),nngₛ⁻¹(om,rwg(x),grid),k,H⃗,grid)[1],
		ω,
		p,
		)






eig_err = ms.M̂ * H⃗[:,1] - ( ω^2 * H⃗[:,1] )
sum(abs2,eig_err)
sum(abs2,H⃗[:,1])
ε⁻¹ = εₛ⁻¹(ω,rwg(p),grid)
nng⁻¹ = nngₛ⁻¹(ω,rwg(p),grid)
k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind=1)
lm = eig_adjt(ms.M̂,ω^2,H⃗[:,1],0.0,H̄)
adj_err = ( (ms.M̂ - (ω^2)*I) * lm ) - ( H̄ - H⃗[:,1] * dot(H⃗[:,1],H̄) )
sum(abs2,adj_err)

lm̄0 = randn(ComplexF64,size(H⃗,1))
lm̄ = lm̄0 ./ dot(lm̄0,lm̄0)
ξ⃗ = linsolve( (ms.M̂ - (ω^2)*I), lm̄ - H⃗[:,1] * dot(H⃗[:,1],lm̄) ; P̂=HelmholtzPreconditioner(ms.M̂) )
adj2_err = ( (ms.M̂ - (ω^2)*I) * ξ⃗ ) - (lm̄ - H⃗[:,1] * dot(H⃗[:,1],lm̄)) #( lm̄ - H⃗[:,1] * dot(H⃗[:,1],lm̄) )
sum(abs2,adj2_err)


sum(eig_adjt(ms.M̂,ω^2,H⃗[:,1],0.0,H̄))
Zygote.gradient(x->abs2(sum(eig_adjt(ms.M̂,x^2,H⃗[:,1],0.0,H̄))),ω)
FiniteDifferences.central_fdm(5,1)(x->abs2(sum(eig_adjt(ms.M̂,x^2,H⃗[:,1],0.0,H̄))),ω)

function foo1(k,p,ω,H⃗,grid)
	ε⁻¹ = εₛ⁻¹(ω,rwg(p),grid)
	nng⁻¹ = nngₛ⁻¹(ω,rwg(p),grid)
	M̂ = HelmholtzMap(real(k),real.(ε⁻¹),grid)
	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
	lm = eig_adjt(
			M̂,								 # Â
			real(ω^2), 							# α
			H⃗[:,1], 					 # x⃗
			0.0, 							# ᾱ
			H̄ ;								 # x̄
			# λ⃗₀=nothing,
			P̂	= HelmholtzPreconditioner(M̂),
		)
	# lm2 = similar(H⃗)
	# solve_adj!(lm2,M̂,H̄,ω^2,H⃗,eigind)
	# println("")
	# println("magmax lm: $(maximum(abs2.(lm)))")
	# println("magmax lm2: $(maximum(abs2.(lm2)))")
	# println("out2: $(sum(abs2.(lm2).^2))")
	return abs2(sum(lm))
end

foo1(k,p,ω,H⃗,grid)
Zygote.gradient(foo1,k,p,ω,H⃗,grid)
Zygote.gradient((a,b,c)->foo1(a,b,c,H⃗,grid),k,p,ω)
Zygote.gradient((a,b,c)->foo1(a,b,c,H⃗,grid),k,p,ω)






FiniteDifferences.grad(central_fdm(9,1),(a,b,c)->foo1(a,b,c,H⃗,grid),k,p,ω)


ε⁻¹ = εₛ⁻¹(ω,rwg(p),grid)
nng⁻¹ = nngₛ⁻¹(ω,rwg(p),grid)
k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
function foo2(k,ω)
	M̂ = HelmholtzMap(k,ε⁻¹,grid)
	lm = eig_adjt(
			M̂,								 # Â
			real(ω^2), 							# α
			H⃗[:,1], 					 # x⃗
			0.0, 							# ᾱ
			H̄ ;								 # x̄
			# λ⃗₀=nothing,
			P̂	= HelmholtzPreconditioner(M̂),
		)
	return abs2(sum(lm))
end
foo2(k,ω)
Zygote.gradient(foo2,k,ω)
FiniteDifferences.grad(central_fdm(9,1),foo2,k,ω)


M̂ = HelmholtzMap(k,ε⁻¹,grid)
function foo3(k,ω)
	lm = eig_adjt(
			M̂,								 # Â
			real(ω^2), 							# α
			H⃗[:,1], 					 # x⃗
			0.0, 							# ᾱ
			H̄ ;								 # x̄
			# λ⃗₀=nothing,
			P̂	= HelmholtzPreconditioner(M̂),
		)
	return abs2(sum(lm))
end
foo3(k,ω)
Zygote.gradient(foo3,k,ω)
FiniteDifferences.grad(central_fdm(9,1),foo3,k,ω)




FiniteDifferences.grad(central_fdm(9,1),(a,b,c)->foo1(a,b,c,H⃗,grid),k,p,ω)


ε⁻¹ = εₛ⁻¹(ω,rwg(p),grid)
nng⁻¹ = nngₛ⁻¹(ω,rwg(p),grid)
M̂ = HelmholtzMap(k,ε⁻¹,grid)
k̄, H̄, nngī  = ∇HMₖH(k,H⃗,nng⁻¹,grid; eigind)
lm = eig_adjt(
		M̂,								 # Â
		ω^2, 							# α
		H⃗[:,1], 					 # x⃗
		0.1, 							# ᾱ
		H̄ )								 # x̄
		# λ⃗₀=nothing,
		# P̂	= HelmholtzPreconditioner(M̂),
	# )

lm2 = eig_adjt(
		M̂,								 # Â
		ω^2, 							# α
		H⃗[:,1], 					 # x⃗
		0.1, 							# ᾱ
		H̄ )

lm3 = eig_adjt(
		M̂,								 # Â
		ω^2, 							# α
		H⃗[:,1], 					 # x⃗
		0.1, 							# ᾱ
		H̄ )

lm4 = linsolve(
	M̂ + (-ω^2*I),
	H̄ - H⃗[:,1] * dot(H⃗[:,1],H̄))

lm5 = linsolve(
	M̂ + (-ω^2*I),
	H̄ - H⃗[:,1] * dot(H⃗[:,1],H̄))


lm2 ≈ lm3
lm4 ≈ lm5

using Zygote: @showgrad
btest = randn(ComplexF64,length(H⃗[:,1]))
function foo2(kk,pp,om)
	ε⁻¹ = εₛ⁻¹(om,rwg(pp),grid)
	M̂ = HelmholtzMap(kk,ε⁻¹,grid)
	# M̂ = HelmholtzMap(@showgrad(kk),@showgrad(ε⁻¹),grid)
	# Â = M̂  - om^2*I
	USM = UniformScalingMap(-(om^2),size(M̂,1))
	Â = M̂ + USM #- ω^2*I
	bt2 = btest - H⃗[:,1] * dot(H⃗[:,1],btest)
	lm = linsolve(
		Â,
		bt2,
	)
	# sum(sin.(abs2.(lm)))
	abs2(sum(lm))
end

foo2(k,p,ω)
k̄_foo2_RAD,p̄_foo2_RAD,om̄_foo2_RAD = Zygote.gradient(foo2,k,p,ω)
k̄_foo2_FD,p̄_foo2_FD,om̄_foo2_FD = FiniteDifferences.grad(central_fdm(9,1),foo2,k,p,ω)


foo2(k,p,ω)
k̄_foo2_RAD,p̄_foo2_RAD,om̄_foo2_RAD = Zygote.gradient(foo2,k,p,ω)
k̄_foo2_FD,p̄_foo2_FD,om̄_foo2_FD = FiniteDifferences.grad(central_fdm(9,1),foo2,k,p,ω)

abs(k̄_foo2_FD - k̄_foo2_RAD) / abs(k̄_foo2_FD)
abs.(p̄_foo2_FD .- p̄_foo2_RAD) ./ abs.(p̄_foo2_FD)
abs(om̄_foo2_FD - om̄_foo2_RAD) / abs(om̄_foo2_FD)

function foo3(kk,pp,om)
	ε⁻¹ = εₛ⁻¹(om,rwg(pp),grid)
	# M̂ = HelmholtzMap(@showgrad(kk),@showgrad(ε⁻¹),grid)
	M̂ = HelmholtzMap(kk,ε⁻¹,grid)
	k̄, H̄, nngī  = ∇HMₖH(k,H⃗,ε⁻¹,grid; eigind=1)
	USM = UniformScalingMap(-om^2,size(M̂,1))
	Â = Zygote.@showgrad(M̂) + USM #- ω^2*I
	lm = linsolve(
		Â,
		H̄ - H⃗[:,1] * dot(H⃗[:,1],H̄),
	)
	sum(sin.(abs2.(lm)))
end

foo3(k,p,ω)
k̄_foo3_RAD,p̄_foo3_RAD,om̄_foo3_RAD = Zygote.gradient(foo3,k,p,ω)


Zygote.gradient(x->sum(reshape(reinterpret(reshape,Float64,reshape(x,)),(3,3,size(x)...))),eic)
##
ω = 0.75
geom = rwg(p)
nng⁻¹, nnginv_pb = Zygote.pullback(nngₛ⁻¹,ω,geom,grid)
ε⁻¹, epsi_pb = Zygote.pullback(εₛ⁻¹,ω,geom,grid)
om̄₁, eī_herm, nngī_herm = ∂²ω²∂k²(ω,ε⁻¹,nng⁻¹,k,H⃗,grid)
om̄₂,geombar_Mₖ,grīd_Mₖ = nnginv_pb(nngī_herm) #nngī2)
om̄₃,geombar_H,grīd_H = epsi_pb(eī_herm) #eī₁)

∂²ω²∂k²_AD
om̄₁
om̄₂
om̄₃

om̄₁ + om̄₂ + om̄₃
om̄₂ + om̄₃
om̄₂ + 0.000663
∂²ω²∂k²_AD - om̄₁
∂²ω²∂k²_AD - ( om̄₁ + om̄₂ + om̄₃ )

##
Â = ms.M̂
α = real(ms.ω²[eigind])
X⃗ = H⃗[:,eigind]
ᾱ = 0
X̄ = Mₖᵀ_plus_Mₖ(H⃗[:,eigind],k,ε⁻¹,grid)
P̂ = HelmholtzPreconditioner(ms.M̂)
λ⃗ = eig_adjt(Â, α, X⃗, ᾱ, X̄)


λ⃗ = eig_adjt(Â, α, X⃗, ᾱ, X̄; P̂)

A = randn(10,10)
A = A + A'
b = randn(10)
x1 = linsolve(A,b)
@assert A * x1 ≈ b

Av = copy(vec(A))
sum(sin.(linsolve(A,b)))
∂A_RAD,∂b_RAD = Zygote.gradient((aa,bb)->sum(sin.(linsolve(aa,bb))),A,b)
∂A_FD,∂b_FD = FiniteDifferences.grad(central_fdm(7,1),(aa,bb)->sum(sin.(linsolve(aa,bb))),A,b)
∂A_err = abs.(∂A_FD .- Matrix(∂A_RAD)) ./ abs.(∂A_FD)
∂b_err = abs.(∂b_FD .- ∂b_RAD) ./ abs.(∂b_FD)

v1 = randn(10)
v2 = randn(10)
v1' * v2
v1 * v2'

outer(v1,v2) = v1 * v2'
delayed_outer = () -> outer(v1,v2)
delayed_outer

using IterativeSolvers
A * x1 - b
x2 = similar(b)

x3,ch = gmres(A,b;verbose=true,log=true,maxiter=1000)
x3,ch = bicgstabl(A,b;verbose=true,log=true,max_mv_products=1000)
A * x3 - b


##

om0 = 0.75
ω = om0
geom = rwg(p)
grid = gr
nnginv,nnginv_pb = Zygote.pullback(nngₛ⁻¹,ω,geom,grid)
epsi,epsi_pb = Zygote.pullback(εₛ⁻¹,ω,geom,grid)

∇HMₖH(k,H⃗,nnginv,grid)
sum(sum.(∇HMₖH(k,H⃗,nnginv,grid)[2]))
Zygote.gradient((a,b,c)->∇HMₖH(a,b,c,grid)[1],k,H⃗,nnginv)
# Zygote.gradient((a,b,c)->sum(∇HMₖH(a,b,c,grid)[2]),k,H⃗,nnginv)
Zygote.gradient((a,b,c)->sum(sum(∇HMₖH(a,b,c,grid)[2])),k,H⃗,nnginv)


d0 = randn(Complex{Float64}, (3,Ns...))
𝓕	 =	plan_fft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place FFT operator 𝓕
𝓕⁻¹ =	plan_bfft(d0,_fftaxes(grid),flags=FFTW.PATIENT) # planned out-of-place iFFT operator 𝓕⁻¹

using StaticArrays: Dynamic
m2 = HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,m⃗))
n2 = HybridArray{Tuple{3,Dynamic(),Dynamic()},Float64}(reinterpret(reshape,Float64,n⃗))
mns1 = mapreduce(x->reshape(flat(x),(1,3,size(x)...)),vcat,(m⃗,n⃗))
mns2 = vcat(reshape(m2,(1,3,Ns...)),reshape(n2,(1,3,Ns...)))
mns1 ≈ mns2

Ninv = 1. / N(grid)
𝓕 * zx_tc( H * Ninv ,mns)
using Tullio
B̄₁ = 𝓕 * kx_tc( conj.(H) ,mns,mag)
B̄₂ = 𝓕 * zx_tc( H * Ninv ,mns)
@tullio B̄[a,b,i,j] := real(B̄₁[a,i,j] * B̄₂[b,i,j])/2 + real(B̄₁[b,i,j] * B̄₂[a,i,j])/2
Bv = reshape(B̄,3,3,128*128)

B̄₁ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  kxtcsp	 *	vec(H),
	𝓕 * kx_tc( conj.(H) ,mns,mag),
	)
B̄₂ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  zxtcsp	 *	vec(H),
	𝓕 * zx_tc( H * Ninv ,mns),
	)
B̄ 	= 	real.( B̄₁  .*  transpose.( B̄₂ ) )

B̄₂ = transpose.(reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  zxtcsp	 *	vec(H),
	𝓕 * zx_tc( H * Ninv ,mns),
	))
B̄ 	= 	real.( B̄₁  .*   B̄₂  )


B̄₁1 = reshape( 𝓕 * kx_tc( conj.(H) ,mns,mag), (3*128,128))
B̄₁2 = reinterpret(
	SVector{3,Complex{Float64}},
	B̄₁1
	)

B̄₂1 = reshape(𝓕 * zx_tc( H * Ninv ,mns), (3*128,128) )
B̄₂2 = reinterpret(
	SVector{3,Complex{Float64}},
	B̄₂1
	)
B̄2 	= 	Hermitian.( real.( B̄₁2  .*  transpose.( B̄₂2 ) ) )

Hsv
H
function foo3(x)
	# B̄₁ = reinterpret(
	# 	reshape,
	# 	SVector{3,Complex{Float64}},
	# 	# 𝓕  *  kxtcsp	 *	vec(H),
	# 	𝓕 * kx_tc( conj.(x) ,mns,mag),
	# 	)
	# B̄₂ = reinterpret(
	# 	reshape,
	# 	SVector{3,Complex{Float64}},
	# 	# 𝓕  *  zxtcsp	 *	vec(H),
	# 	𝓕 * zx_tc( x * Ninv ,mns),
	# 	)
	# B̄ 	= 	real.( B̄₁  .*  transpose.( B̄₂ ) )
	# B̄₁1 = reshape( 𝓕 * kx_tc( conj.(H) ,mns,mag), (3*128,128))
	# B̄₁1 = 𝓕 * kx_tc( conj.(x) ,mns,mag)
	# # B̄₁2 = reinterpret(
	# # 	SVector{3,Complex{Float64}},
	# # 	B̄₁1
	# # 	)
	# B̄₁2 = [ SVector{3,Complex{Float64}}(B̄₁1[1,i,j],B̄₁1[2,i,j],B̄₁1[3,i,j]) for i=1:128,j=1:128]
	#
	#
	# # B̄₂1 = reshape(𝓕 * zx_tc( H * Ninv ,mns), (3*128,128) )
	# B̄₂1 = 𝓕 * zx_tc( x * Ninv ,mns)
	# # B̄₂2 = reinterpret(
	# # 	SVector{3,Complex{Float64}},
	# # 	B̄₂1
	# # 	)
	# B̄₂2 = [ SVector{3,Complex{Float64}}(B̄₂1[1,i,j],B̄₂1[2,i,j],B̄₂1[3,i,j]) for i=1:128,j=1:128]
	# B̄ 	= 	Hermitian.( real.( B̄₁2  .*  transpose.( B̄₂2 ) ) )
	B̄₁4 = 𝓕 * kx_tc( conj.(x) ,mns,mag)
	B̄₂4 = 𝓕 * zx_tc( x * Ninv ,mns)
	@tullio B̄[a,b,i,j] := real(B̄₁4[a,i,j] * B̄₂4[b,i,j])
	# return B̄
	return reinterpret(SMatrix{3,3,Float64,9},reshape(B̄,9*128,128))
end

function foo4(A::AbstractArray{SMatrix{3,3,Float64,9}})
	sum(sum.(A))
end

Hc = copy(H)
foo3(Hc)
reinterpret(reshape,SMatrix{3,3,Float64,9},foo3(Hc))
foo4(copy(foo3(Hc)))
Zygote.gradient(x->foo4(foo3(x)),Hc)

B̄₁2 = vec( 𝓕 * kx_tc( conj.(H) ,mns,mag) )
B̄₂2 = vec( 𝓕 * zx_tc( H * Ninv ,mns) )

B̄₁3 = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  kxtcsp	 *	vec(H),
	𝓕 * kx_tc( conj.(H) ,mns,mag),
	) |> copy
B̄₂3 = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  zxtcsp	 *	vec(H),
	𝓕 * zx_tc( H * Ninv ,mns),
	) |> copy
B̄3 	= 	real.( B̄₁3  .*  transpose.( B̄₂3 ) )
using Tullio
B̄₁4 = 𝓕 * kx_tc( conj.(H) ,mns,mag)
B̄₂4 = 𝓕 * zx_tc( H * Ninv ,mns)
@tullio B̄4[a,b,i,j] := real(B̄₁4[a,i,j] * B̄₂4[b,i,j])
reinterpret(reshape,SMatrix{3,3,Float64,9},B̄4)
reinterpret(SMatrix{3,3,Float64,9},copy(reshape(B̄4,(9,128,128))))
Bb41 = reinterpret(reshape,SMatrix{3,3,Float64,9},copy(reshape(B̄4,(9,128,128))))
Bb42 = reinterpret(SMatrix{3,3,Float64,9},reshape(B̄4,(9*128,128)))
Bb41 ≈ Bb42
Bb43 = [SMatrix{3,3,Float64,9}(B̄4[:,:,i,j]) for i=1:128,j=1:128]
Bb41 ≈ Bb43
unflat(B̄4)
##
om0 = 0.75
M̂ = ms.M̂
kxtcsp 	= kx_tc_sp(k,gr)
zxtcsp 	= zx_tc_sp(k,gr)
eisp 	= ε⁻¹_sp(om0,rwg(p),gr)
nngsp 	= nng⁻¹_sp(om0,rwg(p),gr)
𝓕 = LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(fft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
𝓕⁻¹ = LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(ifft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
𝓕⁻¹b = LinearMap{ComplexF64}(d::AbstractVector{ComplexF64} -> vec(bfft(reshape(d,(3,Ns...)),(2:3)))::AbstractVector{ComplexF64},*(3,Ns...),ishermitian=true,ismutating=false)
Hsv = reinterpret(reshape, SVector{2,Complex{Float64}}, H )
A_sp 	=	-transpose(kxtcsp)
B_sp 	=	𝓕⁻¹b	*	nngsp	*	𝓕
C_sp	=	zxtcsp

zxtc_to_mn = SMatrix{3,3}(	[	0 	-1	  0
								1 	 0	  0
								0 	 0	  0	  ]	)

kxtc_to_mn = SMatrix{2,2}(	[	0 	-1
								1 	 0	  ]	)



Ā₁		=	conj.(Hsv)
Ā₂ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# reshape(
	# 	𝓕⁻¹ * nngsp * 𝓕 * zxtcsp * vec(H),
	# 	(3,size(gr)...),
	# 	),
	M̂.𝓕⁻¹ * ε⁻¹_dot(  M̂.𝓕 * zx_tc(H * M̂.Ninv,mns) , real(flat(nnginv))),
	)
Ā 	= 	Ā₁  .*  transpose.( Ā₂ )
m̄n̄_Ā = transpose.( (kxtc_to_mn,) .* real.(Ā) )
m̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (1,) )
n̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (2,) )
māg_Ā = dot.(n⃗, n̄_Ā) + dot.(m⃗, m̄_Ā)
k̄_Mₖ_Ā = mag_m_n_pb( ( māg_Ā, m̄_Ā.*mag, n̄_Ā.*mag ) )[1]

B̄₁ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  kxtcsp	 *	vec(H),
	M̂.𝓕 * kx_tc( conj.(H) ,mns,mag),
	)
B̄₂ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# 𝓕  *  zxtcsp	 *	vec(H),
	M̂.𝓕 * zx_tc( H * M̂.Ninv ,mns),
	)
B̄ 	= 	real.( B̄₁  .*  transpose.( B̄₂ ) )


C̄₁ = reinterpret(
	reshape,
	SVector{3,Complex{Float64}},
	# reshape(
	# 	𝓕⁻¹ * nngsp * 𝓕 * kxtcsp * -vec(H),
	# 	(3,size(gr)...),
	# 	),
	M̂.𝓕⁻¹ * ε⁻¹_dot(  M̂.𝓕 * -kx_tc(H* M̂.Ninv,mns,mag) , real(flat(nnginv))),
	)
C̄₂ =   conj.(Hsv)
C̄ 	= 	C̄₁  .*  transpose.( C̄₂ )
m̄n̄_C̄ = 			 (zxtc_to_mn,) .* real.(C̄)
m̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (1,) )
n̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (2,) )
k̄_Mₖ_C̄ = mag_m_n_pb( ( nothing, m̄_C̄, n̄_C̄ ) )[1]

nngī_Mₖ = ( B̄ .+ transpose.(B̄) ) ./ 2
nngī_Mₖ_magmax = maximum(abs.(flat(nngī_Mₖ)))
k̄_Mₖ = k̄_Mₖ_Ā + k̄_Mₖ_C̄

println("")
println("magmax(nngī_Mₖ) = $(nngī_Mₖ_magmax)")
println("k̄_Mₖ = $k̄_Mₖ")

# @btime begin
# 	C̄ = 	reinterpret(reshape, SVector{3,Complex{Float64}}, reshape( 𝓕⁻¹ * nngsp * 𝓕 * kxtcsp * -vec(H), (3,size(gr)...)) )  .*  transpose.( conj.(Hsv) )
# 	m̄n̄_C̄ = 			 (zxtc_to_mn,) .* real.(C̄)
# 	m̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (1,) )
# 	n̄_C̄ = 		view.( m̄n̄_C̄, (1:3,), (2,) )
# 	k̄_Mₖ_C̄ = mag_m_n_pb( ( nothing, m̄_C̄, n̄_C̄ ) )[1]
#
# 	Ā = 	conj.(Hsv)   .*  transpose.( reinterpret(reshape, SVector{3,Complex{Float64}}, reshape( 𝓕⁻¹ * nngsp * 𝓕 * zxtcsp * vec(H), (3,size(gr)...)) ) )
# 	m̄n̄_Ā = transpose.( (kxtc_to_mn,) .* real.(Ā) )
# 	m̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (1,) )
# 	n̄_Ā = 		view.( m̄n̄_Ā, (1:3,), (2,) )
# 	māg_Ā = dot.(n⃗, n̄_Ā) + dot.(m⃗, m̄_Ā)
# end
# 2.022 s (1683353 allocations: 4.09 GiB)
# @btime mag_m_n_pb( ( māg_Ā, m̄_Ā.*mag, n̄_Ā.*mag ) )
# 1.932 s (1650232 allocations: 4.06 GiB)
##
g⃗s = collect(g⃗(gr))
(mag, m⃗, n⃗), mag_m_n_pb = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
m = M̂.m
n = M̂.n
# HMₖH, HMₖH_pb = Zygote.pullback(H_Mₖ_H,H,ε⁻¹,mag,m,n)
HMₖH, HMₖH_pb = Zygote.pullback(H_Mₖ_H,H,nnginv,mag,m,n)
# @btime HMₖH_pb(1) # 4.553 ms (237 allocations: 15.89 MiB)
H̄2, eī2, māg2,m̄2,n̄2 = HMₖH_pb(1)
m̄v2 = copy(reinterpret(reshape,SVector{3,Float64},real(m̄2)))
n̄v2 = copy(reinterpret(reshape,SVector{3,Float64},real(n̄2)))
k̄_Mₖ_AD = mag_m_n_pb( (real(māg2), m̄v2, n̄v2) )[1]

nngī_Mₖ_AD_magmax = maximum(abs.(flat(eī2)))
println("magmax(nngī_Mₖ_AD) = $(nngī_Mₖ_AD_magmax)")
println("magmax(nngī_Mₖ)_err = $( abs( nngī_Mₖ_magmax - nngī_Mₖ_AD_magmax ) / abs(nngī_Mₖ_AD_magmax) )")

println("k̄_Mₖ_AD = $k̄_Mₖ_AD")
println("k̄_Mₖ_err = $( abs( k̄_Mₖ - k̄_Mₖ_AD ) / abs(k̄_Mₖ_AD) )")
##

māg_Ā
k̄_Mₖ_Ā
mag
māg_A_man = (k̄_Mₖ_Ā / k) .* mag
k
ẑ = SVector(0,0,1)
k⃗ = SVector(0,0,k)
kp⃗g = (k⃗,) .- g⃗s
kp̂g = kp⃗g ./ mag
kp⃗gxz = cross.(kp⃗g,(ẑ,))
kp̂gxz = cross.(kp̂g,(ẑ,))
mxkp⃗gxz = cross.(m⃗,kp⃗gxz)
nxkp⃗gxz = cross.(n⃗,kp⃗gxz)
mxkp̂gxz = cross.(m⃗,kp̂gxz)
nxkp̂gxz = cross.(n⃗,kp̂gxz)


using Zygote: Buffer, dropgrad
function mag_m_n3(k⃗::SVector{3,T},grid::Grid) where T <: Real
	local ẑ = SVector(0.,0.,1.)
	local ŷ = SVector(0.,1.,0.)
	g⃗s = g⃗(dropgrad(grid))
	n = Buffer(g⃗s,size(g⃗s))
	m = Buffer(g⃗s,size(g⃗s))
	mag = Buffer(zeros(T,size(g⃗s)),size(g⃗s))
	@fastmath @inbounds for i ∈ eachindex(g⃗s)
		@inbounds kpg::SVector{3,T} = k⃗ - g⃗s[i]
		@inbounds mag[i] = norm(kpg)
		@inbounds n[i] =   ( ( abs2(kpg[1]) + abs2(kpg[2]) ) > 0. ) ?  normalize( cross( ẑ, kpg ) ) : ŷ
		@inbounds m[i] =  normalize( cross( n[i], kpg )  )
	end
	return copy(mag), copy(m), copy(n)
end

mag3,m3,n3 = mag_m_n3(k⃗,gr)

(mag, m⃗, n⃗), mag_m_n_pb5 = Zygote.pullback(x->mag_m_n(x,g⃗s),k)
mag_m_n4(k⃗,g⃗s) .≈ mag_m_n(k⃗,g⃗s)
(mag4,m4,n4), mag_m_n4_pb = pullback(x->mag_m_n4(SVector(0.,0.,x),g⃗s),k)
(mag4,m4,n4), mag_m_n4_pb = Zygote.pullback(mag_m_n4,k⃗,g⃗s)
mag_m_n_pb((māg_Ā,mag.*m̄_Ā,n̄_Ā))[1]
mag_m_n4_pb((māg_Ā,mag.*m̄_Ā,n̄_Ā))[1]

ΔmagmnA = (māg_Ā,mag.*m̄_Ā,n̄_Ā)
@btime mag_m_n_pb5($ΔmagmnA)[1]
@btime mag_m_n4_pb($ΔmagmnA)[1]
@btime mag_m_n($k⃗,$g⃗s)
∇ₖmag_m_n((māg_Ā,mag.*m̄_Ā,n̄_Ā),(mag,m⃗,n⃗);dk̂=SVector(0.,0.,1.))
∇ₖmag_m_n(māg_Ā,mag.*m̄_Ā,n̄_Ā,mag,m⃗,n⃗;dk̂=SVector(0.,0.,1.))

Zygote.gradient(∇ₖmag_m_n,māg_Ā,mag.*m̄_Ā,n̄_Ā,mag,m⃗,n⃗)


mag3 ≈ mag
m3 ≈ m⃗
n3 ≈ n⃗

k̄_Ā_mag_man = dot(vec(māg_Ā),inv.(vec(mag))) * k
k̄_Ā_m_man = sum( dot.( m̄_Ā .* mag , cross.(m⃗, cross.(kp⃗g, (ẑ,) ) ) ./ mag.^2 ) )
k̄_Ā_n_man = sum( dot.( n̄_Ā , cross.(n⃗, cross.(kp⃗g, (ẑ,) ) ) ./ mag.^2 ) )
k̄_Ā_man = k̄_Ā_mag_man + k̄_Ā_m_man + k̄_Ā_n_man
k̄_Ā_man / k̄_Mₖ_Ā

function ∇ₖmag_m_n(māg,m̄,n̄,mag,m⃗,n⃗;dk̂=ẑ)
	kp̂g_over_mag = cross.(m⃗,n⃗)./mag
	k̄_mag = sum( māg .* dot.( kp̂g_over_mag, (dk̂,) ) .* mag )
	k̄_m = -sum( dot.( m̄ , cross.(m⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	k̄_n = -sum( dot.( n̄ , cross.(n⃗, cross.( kp̂g_over_mag, (dk̂,) ) ) ) )
	return +( k̄_mag, k̄_m, k̄_n )
end

kp⃗g1 = fill(k⃗,size(gr)...) - g⃗(dropgrad(gr))
kp⃗g2 = cross.(m⃗,n⃗).*mag

kp⃗g1 ≈ kp⃗g2
kp⃗g ≈ kp⃗g2

∇ₖmag_m_n(māg_Ā,mag.*m̄_Ā,n̄_Ā,mag,m⃗,n⃗;dk̂=ẑ)

∇ₖmag_m_n(māg_Ā,mag.*m̄_Ā,n̄_Ā,mag,m⃗,n⃗;dk̂=ẑ)

g⃗s = g⃗(dropgrad(grid))


foo1(x) = sum(sin.(x))
foo1_mag, foo1_mag_pb = Zygote.pullback(foo1,mag)
māg_foo1 = foo1_mag_pb(1)[1]
m̄v_foo1 = nothing
n̄v_foo1 = nothing
k̄_foo1 = mag_m_n_pb((māg_foo1,m̄v_foo1,n̄v_foo1))[1]
k̄_foo1_man = dot(vec(māg_foo1),inv.(vec(mag))) * k
k̄_foo1 / k̄_foo1_man

foo2(x) = sum(sin.(vec(flat(x))))
foo2_m, foo2_m_pb = Zygote.pullback(foo2,m⃗)
m̄_foo2 = foo2_m_pb(1)[1]
māg_foo2 = nothing
n̄_foo2 = nothing
k̄_foo2_m = mag_m_n_pb((māg_foo2,m̄_foo2,n̄_foo2))[1]
k̄_foo2_m_man = sum( dot.( m̄_foo2 , cross.(m⃗, cross.(kp⃗g, (ẑ,) ) ) ./ mag.^2 ) )

foo2_n, foo2_n_pb = Zygote.pullback(foo2,n⃗)
n̄_foo2_n = foo2_n_pb(1)[1]
k̄_foo2_n = mag_m_n_pb((nothing,nothing,n̄_foo2_n))[1]
k̄_foo2_n_man = sum( dot.( n̄_foo2_n , cross.(n⃗, cross.(kp⃗g, (ẑ,) ) ) ./ mag.^2 ) )



abs.(flat(kp⃗gxz)) |> maximum
abs.(flat(mxkp⃗gxz)) |> maximum
abs.(flat(nxkp⃗gxz)) |> maximum

abs.(flat(kp̂gxz)) |> maximum
abs.(flat(mxkp̂gxz)) |> maximum
abs.(flat(nxkp̂gxz)) |> maximum
abs.(flat(n⃗)) |> maximum

flat(dm3) ./ flat(mxkp̂gxz)
( flat(mxkp̂gxz ./ mag )  ) ./ flat(dm3)
( flat(mxkp⃗gxz ./ mag.^2 )  ) ./ flat(dm3)

# ( flat(mxkp⃗gxz )  ) ./ flat(dm3)

kp̂gxz ≈ n⃗
kp⃗gxz ≈ -n⃗
# k̄_foo1_man = dot(vec(māg_foo1),inv.(vec(mag))) * k
# k̄_foo1 / k̄_foo1_man

function dmagmn_dk_FD(k0,dk)
	mag0,m0,n0 = mag_m_n(k0-dk/2,g⃗s)
	mag1,m1,n1 = mag_m_n(k0+dk/2,g⃗s)
	dmag = ( mag1 .- mag0 ) ./ dk
	dm = ( m1 .- m0 ) ./ dk
	dn = ( n1 .- n0 ) ./ dk
	return dmag, dm, dn
end

dmag1,dm1,dn1 = dmagmn_dk_FD(k,1e-3)
dmag2,dm2,dn2 = dmagmn_dk_FD(k,1e-5)
dmag3,dm3,dn3 = dmagmn_dk_FD(k,1e-7)

dmag3 ≈ dmag2
dmag3

##
m̄2r = real(m̄2)
n̄2r = real(n̄2)
m̄f = copy(flat(SVector{3}.(m̄)))
m̄mf = copy(flat(SVector{3}.(m̄).*mag))
n̄f = copy(flat(SVector{3}.(n̄)))
n̄mf = copy(flat(SVector{3}.(n̄).*mag))

m̄2r ./ m̄f
m̄2r ./ m̄mf
n̄2r ./ n̄f
n̄2r ./ n̄mf

@show maximum(abs.(m̄2r))
@show maximum(abs.(m̄f))
@show maximum(abs.(m̄mf))
@show maximum(abs.(n̄2r))
@show maximum(abs.(n̄f))
@show maximum(abs.(n̄mf))


## single ω solve_n gradient checks, ms created within solve_n
function gradtest_solve_n(ω0)
        err_style = NEGATIVE*BOLD*BLUE_FG      # defined in Crayons.Box
        println("...............................................................")
        println("solve_n (single ω) gradient checks, ms created within solve_n: ")
        @show ω0
        neff1,ng1 = solve_n(ω0+rand()*0.1,rwg(p),gr)
        neff2,ng2 = solve_n(ω0+rand()*0.1,rwg2(p),gr)

        println("∂n_om, non-dispersive materials:")
        om = ω0 #+rand()*0.1
        println("\t∂n_om (Zygote):")
        ∂n_om_RAD = Zygote.gradient(x->solve_n(x,rwg2(p),gr)[1],om)[1]
        println("\t$∂n_om_RAD")
        solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂n_om (FD):")
        ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(x,rwg2(p),gr)[1],om)[1]
        println("\t$∂n_om_FD")
        println(err_style("∂n_om_err:"))
        ∂n_om_err = abs(∂n_om_RAD - ∂n_om_FD) / abs(∂n_om_FD)
        println("$∂n_om_err")
        n_ndisp = solve_n(om,rwg2(p),gr)[1]
        ng_manual_ndisp = n_ndisp + om * ∂n_om_FD
        println("ng_manual: $ng_manual_ndisp")

        println("∂ng_om, non-dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂ng_om (Zygote):")
        ∂ng_om_RAD = Zygote.gradient(x->solve_n(x,rwg2(p),gr)[2],om)[1]
        println("\t$∂ng_om_RAD")
        solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂ng_om (FD):")
        ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(x,rwg2(p),gr)[2],om)[1]
        println("\t$∂ng_om_FD")
        println(err_style("∂ng_om_err:"))
        ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)
        println("$∂ng_om_err")

        println("∂n_om, dispersive materials:")
        om = ω0 #+rand()*0.1
        println("\t∂n_om (Zygote):")
        ∂n_om_RAD = Zygote.gradient(x->solve_n(x,rwg(p),gr)[1],om)[1]
        println("\t$∂n_om_RAD")
        solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂n_om (FD):")
        ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(x,rwg(p),gr)[1],om)[1]
        println("\t$∂n_om_FD")
        println(err_style("∂n_om_err:"))
        ∂n_om_err = abs(∂n_om_RAD - ∂n_om_FD) / abs(∂n_om_FD)
        println("$∂n_om_err")
        n_disp = solve_n(om,rwg(p),gr)[1]
        ng_manual_disp = n_disp + om * ∂n_om_FD
        println("ng_manual: $ng_manual_disp")

        println("∂ng_om, dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂ng_om (Zygote):")
        ∂ng_om_RAD = Zygote.gradient(x->solve_n(x,rwg(p),gr)[2],om)[1]
        println("\t$∂ng_om_RAD")
        solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂ng_om (FD):")
        ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(x,rwg(p),gr)[2],om)[1]
        println("\t$∂ng_om_FD")
        println(err_style("∂ng_om_err:"))
        ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs.(∂ng_om_FD)
        println("$∂ng_om_err")

        println("∂n_p, non-dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂n_p (Zygote):")
        ∂n_p_RAD =  Zygote.gradient(x->solve_n(om,rwg2(x),gr)[1],p)[1]
        println("\t$∂n_p_RAD")
        solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂n_p (FD):")
        ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(om,rwg2(x),gr)[1],p)[1]
        println("\t$∂n_p_FD")
        println(err_style("∂n_p_err:"))
        ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)
        println("$∂n_p_err")

        println("∂n_p, dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂n_p (Zygote):")
        ∂n_p_RAD =  Zygote.gradient(x->solve_n(om,rwg(x),gr)[1],p)[1]
        println("\t$∂n_p_RAD")
        solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂n_p (FD):")
        ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(om,rwg(x),gr)[1],p)[1]
        println("\t$∂n_p_FD")
        println(err_style("∂n_p_err:"))
        ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)
        println("$∂n_p_err")

        println("∂ng_p, non-dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂ng_p (Zygote):")
        ∂ng_p_RAD = Zygote.gradient(x->solve_n(om,rwg2(x),gr)[2],p)[1]
        println("\t$∂ng_p_RAD")
        solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂ng_p (FD):")
        ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(om,rwg2(x),gr)[2],p)[1]
        println("\t$∂ng_p_FD")
        println(err_style("∂ng_p_err:"))
        ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD
        println("$∂ng_p_err")

        println("∂ng_p, dispersive materials:")
        # om = ω0+rand()*0.1
        println("\t∂ng_p (Zygote):")
        ∂ng_p_RAD = Zygote.gradient(x->solve_n(om,rwg(x),gr)[2],p)[1]
        println("\t$∂ng_p_RAD")
        solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂ng_p (FD):")
        ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(om,rwg(x),gr)[2],p)[1]
        println("\t$∂ng_p_FD")
        println(err_style("∂ng_p_err:"))
        ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD
                println("$∂ng_p_err")
                println("...............................................................")
end

gradtest_solve_n(0.7)
gradtest_solve_n(0.8)
gradtest_solve_n(0.9)
##

## single ω solve_n gradient checks, ms created within solve_n
function gradtest_solve_n_sweep(ω0;om_grads=false)
        println("...............................................................")
        println("ω sweep solve_n gradient checks, ms created within solve_n: ")
        @show ω0
        neff1,ng1 = solve_n(ω0.+rand()*0.1,rwg(p),gr)
        neff2,ng2 = solve_n(ω0.+rand()*0.1,rwg2(p),gr)

        if om_grads
                println("∂n_om, non-dispersive materials:")
                om = ω0.+rand()*0.1
                println("\t∂n_om (Zygote):")
                ∂n_om_RAD = Zygote.gradient(x->sum(solve_n(x,rwg2(p),gr)[1]),om)[1]
                println("\t$∂n_om_RAD")
                # solve_n(om+rand()*0.2,rwg(p),gr)
                println("\t∂n_om (FD):")
                ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(x,rwg2(p),gr)[1]),om)[1]
                println("\t$∂n_om_FD")
                @show ∂n_om_err = abs.(∂n_om_RAD .- ∂n_om_FD) ./ abs.(∂n_om_FD)

                println("∂ng_om, non-dispersive materials:")
                om = ω0.+rand()*0.1
                println("\t∂ng_om (Zygote):")
                ∂ng_om_RAD = Zygote.gradient(x->sum(solve_n(x,rwg2(p),gr)[2]),om)[1]
                println("\t$∂ng_om_RAD")
                # solve_n(om+rand()*0.2,rwg(p),gr)
                println("\t∂ng_om (FD):")
                ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(x,rwg2(p),gr)[2]),om)[1]
                println("\t$∂ng_om_FD")
                @show ∂ng_om_err = abs.( ∂ng_om_RAD .-  ∂ng_om_FD) ./  abs.(∂ng_om_FD)

                println("∂ng_om, dispersive materials:")
                om = ω0.+rand()*0.1
                println("\t∂ng_om (Zygote):")
                ∂ng_om_RAD = Zygote.gradient(x->sum(solve_n(x,rwg(p),gr)[2]),om)[1]
                println("\t$∂ng_om_RAD")
                # solve_n(om+rand()*0.2,rwg(p),gr)
                println("\t∂ng_om (FD):")
                ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(x,rwg(p),gr)[2]),om)[1]
                println("\t$∂ng_om_FD")
                @show ∂ng_om_err = abs.( ∂ng_om_RAD .-  ∂ng_om_FD) ./  abs.(∂ng_om_FD)
        end

        println("∂n_p, non-dispersive materials:")
        om = ω0.+rand()*0.1
        println("\t∂n_p (Zygote):")
        ∂n_p_RAD =  Zygote.gradient(x->sum(solve_n(om,rwg2(x),gr)[1]),p)[1]
        println("\t$∂n_p_RAD")
        # solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂n_p (FD):")
        ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(om,rwg2(x),gr)[1]),p)[1]
        println("\t$∂n_p_FD")
        @show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

        println("∂n_p, dispersive materials:")
        om = ω0.+rand()*0.1
        println("\t∂n_p (Zygote):")
        ∂n_p_RAD =  Zygote.gradient(x->sum(solve_n(om,rwg(x),gr)[1]),p)[1]
        println("\t$∂n_p_RAD")
        # solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂n_p (FD):")
        ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(om,rwg(x),gr)[1]),p)[1]
        println("\t$∂n_p_FD")
        @show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

        println("∂ng_p, non-dispersive materials:")
        om = ω0.+rand()*0.1
        println("\t∂ng_p (Zygote):")
        ∂ng_p_RAD = Zygote.gradient(x->sum(solve_n(om,rwg2(x),gr)[2]),p)[1]
        println("\t$∂ng_p_RAD")
        # solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂ng_p (FD):")
        ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(om,rwg2(x),gr)[2]),p)[1]
        println("\t$∂ng_p_FD")
        @show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD

        println("∂ng_p, dispersive materials:")
        om = ω0.+rand()*0.1
        println("\t∂ng_p (Zygote):")
        ∂ng_p_RAD = Zygote.gradient(x->sum(solve_n(om,rwg(x),gr)[2]),p)[1]
        println("\t$∂ng_p_RAD")
        # solve_n(om+rand()*0.2,rwg(p),gr)
        println("\t∂ng_p (FD):")
        ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(om,rwg(x),gr)[2]),p)[1]
        println("\t$∂ng_p_FD")
        @show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD
        println("...............................................................")
end

gradtest_solve_n_sweep([0.65, 0.75, 0.85])
gradtest_solve_n_sweep(collect(0.55:0.03:0.85))

##
ns,ngs = solve_n(ms,ωs,rwg(p))

##

@show ∂sumng_RAD = Zygote.gradient(x->sum(solve_n(ms,[0.6,0.7],rwg(x))[2]),p)[1]




@show ∂sumng_FD = FiniteDifferences.grad(central_fdm(3,1),x->sum(solve_n(ms,[0.6,0.7],rwg(x))[2]),p)[1]
@show ∂sumng_err = abs.(∂sumng_RAD[2] .- ∂sumng_FD) ./ abs.(∂sumng_FD)


##
λs = 1 ./ ωs
fig,ax,sc1 = scatter(λs,ng1,color=logocolors[:red])
lines!(ax,λs,ng1,color=logocolors[:red],lw=2)
lines!(ax,λs,n1,color=logocolors[:blue],lw=2)
scatter!(ax,λs,n1,color=logocolors[:blue])
fig
##

solve_n(ms,ωs,rwg(p))
solve_n(ωs,rwg(p),gr)

function var_ng(ωs,p)
    ngs = solve_n(ωs,rwg(p),gr)[2]
    # mean(  ngs.^2  ) - mean(ngs)^2
    var(real(ngs))
end

var_ng(ωs,p)
@show ∂vng_RAD = Zygote.gradient(var_ng,ωs,p)
@show ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)[1]
@show ∂vng_err = abs.(∂vng_RAD[2] .- ∂vng_FD) ./ abs.(∂vng_FD)

ωs = collect(0.55:0.03:0.85)
@show ∂vng_RAD = Zygote.gradient(var_ng,ωs,p)
@show ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)[1]
@show ∂vng_err = abs.(∂vng_RAD[2] .- ∂vng_FD) ./ abs.(∂vng_FD)


@show ∂sumng_RAD = Zygote.gradient(x->sum([solve_n(ms,om,rwg(x))[2] for om in [0.6,0.7] ]),p)[1]
@show ∂sumng_FD = FiniteDifferences.grad(central_fdm(3,1),x->sum([solve_n(ms,om,rwg(x))[2] for om in [0.6,0.7] ]),p)[1]
@show ∂sumng_err = abs.(∂sumng_RAD[2] .- ∂sumng_FD) ./ abs.(∂sumng_FD)

# @time ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)
#
# Zygote.gradient((x,y)->solve_n(ms,x,rwg(y))[2],1/0.85,p)
# Zygote.gradient(ωs,p) do oms,x
# 	ngs = solve_n(Zygote.dropgrad(ms),oms,rwg(x))[2]
#     mean( abs2.( ngs ) ) - abs2(mean(ngs))
# end

## Define with constant indices


##
fig,ax,sc1 = scatter(λs,ng2,color=logocolors[:red])
lines!(ax,λs,ng2,color=logocolors[:red],lw=2)
lines!(ax,λs,n2,color=logocolors[:blue],lw=2)
scatter!(ax,λs,n2,color=logocolors[:blue])
fig
##
function var_ng2(ωs,p)
    ngs = solve_n(Zygote.dropgrad(ms),ωs,rwg2(p))[2]
    mean( abs2.( ngs ) ) - abs2(mean(ngs))
end
var_ng2(ωs,p)

@show ∂vng2_RAD = Zygote.gradient(var_ng2,ωs,p)
@show ∂vng2_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng2(ωs,x),p)[1]
@show ∂vng2_err = abs.(∂vng2_RAD[2] .- ∂vng2_FD) ./ abs.(∂vng2_FD)


var_ng(ωs,p)
@show ∂vng_RAD = Zygote.gradient(var_ng,ωs,p)
@show ∂vng_FD = FiniteDifferences.grad(central_fdm(3,1),x->var_ng(ωs,x),p)[1]
@show ∂vng_err = abs.(∂vng_RAD[2] .- ∂vng_FD) ./ abs.(∂vng_FD)

##

∂n_RAD = zeros(length(ωs),3)
∂n_FD = zeros(length(ωs),3)
∂n_err = zeros(length(ωs),3)

∂ng_RAD = zeros(length(ωs),3)
∂ng_FD = zeros(length(ωs),3)
∂ng_err = zeros(length(ωs),3)

for omind in 1:length(ωs)
    ∂n_RAD[omind,:] = Zygote.gradient(x->solve_n(ms,ωs[omind],rwg2(x))[1],p)[1]
    ∂ng_RAD[omind,:] = Zygote.gradient(x->solve_n(ms,ωs[omind],rwg2(x))[2],p)[1]
    ∂n_FD[omind,:] = FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,ωs[omind],rwg2(x))[1],p)[1]
    ∂ng_FD[omind,:] = FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,ωs[omind],rwg2(x))[2],p)[1]
end
∂n_err = abs.(∂n_RAD .- ∂n_FD) ./ abs.(∂n_FD)
∂ng_err = abs.(∂ng_RAD .- ∂ng_FD) ./ abs.(∂ng_FD)

##
ln = lines(collect(λs),∂n_err[:,1],color=logocolors[:green])
lines!(collect(λs),∂n_err[:,2],color=logocolors[:blue])
lines!(collect(λs),∂n_err[:,3],color=logocolors[:red])

lng = lines(collect(λs),∂ng_err[:,1],color=logocolors[:green])
lines!(collect(λs),∂ng_err[:,2],color=logocolors[:blue])
lines!(collect(λs),∂ng_err[:,3],color=logocolors[:red])

##
ei2 = εₛ⁻¹(1/1.55,rwg2(p);ms)
Zygote.gradient((x,y)->sum(sum(εₛ⁻¹(x,rwg2(y);ms))),1/1.55,p)



@time ∂sumei_FD = FiniteDifferences.grad(central_fdm(3,1),x->sum(sum(εₛ⁻¹(1/1.55,rwg2(x);ms))),p)




Zygote.gradient((x,y)->solve_n(ms,x,rwg2(y))[1],1/1.55,p)




@time ∂n2_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,1/1.55,rwg2(x))[1],p)





Zygote.gradient((x,y)->solve_n(ms,x,rwg2(y))[2],1/1.55,p)




@time ∂ng2_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,1/1.55,rwg2(x))[2],p)





omsq2,H2 = solve_ω²(ms,1.45,rwg2(p))
summag4(HH) = sum(abs2.(HH).^2)

@show ∂omsq_k_RAD = Zygote.gradient(x->solve_ω²(ms,x,rwg2(p))[1],1.45)[1]
@show ∂omsq_k_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,x,rwg2(p))[1],1.45)[1]
@show ∂omsq_k_err = abs(∂omsq_k_RAD - ∂omsq_k_FD) / ∂omsq_k_FD

@show ∂sm4_k_RAD = Zygote.gradient(x->summag4(solve_ω²(ms,x,rwg2(p))[2]),1.45)[1]
@show ∂sm4_k_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_ω²(ms,x,rwg2(p))[2]),1.45)[1]
@show ∂omsq_k_err = abs( ∂sm4_k_RAD -  ∂sm4_k_FD) /  ∂sm4_k_FD

@show ∂omsq_p_RAD =  Zygote.gradient(x->solve_ω²(ms,1.45,rwg2(x))[1],p)[1]
@show ∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg2(x))[1],p)[1]
@show ∂omsq_p_err = abs.(∂omsq_p_RAD .- ∂omsq_p_FD) ./ ∂omsq_p_FD

# @show ∂omsq_p_RAD =  Zygote.gradient(x->solve_ω²(ms,1.45,rwg(x))[1],p)[1]
# @show ∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg(x))[1],p)[1]
# @show ∂omsq_p_err = abs.(∂omsq_p_RAD .- ∂omsq_p_FD) ./ ∂omsq_p_FD

@show ∂sm4_p_RAD = Zygote.gradient(x->summag4(solve_ω²(ms,1.45,rwg2(x))[2]),p)[1]
@show ∂sm4_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_ω²(ms,1.45,rwg2(x))[2]),p)[1]
@show ∂sm4_p_err = abs.(∂sm4_p_RAD .- ∂sm4_p_FD) ./ ∂sm4_p_FD

k2,H22 = solve_k(ms,0.7,rwg2(p))

@show ∂k_om_RAD = Zygote.gradient(x->solve_k(ms,x,rwg2(p))[1],0.7)[1]
@show ∂k_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,x,rwg2(p))[1],0.7)[1]
@show ∂k_om_err = abs(∂k_om_RAD - ∂k_om_FD) / abs(∂k_om_FD)

@show ∂sm4_om_RAD = Zygote.gradient(x->summag4(solve_k(ms,x,rwg2(p))[2]),0.7)[1]
@show ∂sm4_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_k(ms,x,rwg2(p))[2]),0.7)[1]
@show ∂sm4_om_err = abs( ∂sm4_om_RAD -  ∂sm4_om_FD) /  abs(∂sm4_om_FD)

@show ∂k_p_RAD =  Zygote.gradient(x->solve_k(ms,0.7,rwg2(x))[1],p)[1]
@show ∂k_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.7,rwg2(x))[1],p)[1]
@show ∂k_p_err = abs.(∂k_p_RAD .- ∂k_p_FD) ./ abs.(∂k_p_FD)

@show ∂k_p_RAD =  Zygote.gradient(x->solve_k(ms,0.7,rwg(x))[1],p)[1]
@show ∂k_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.7,rwg(x))[1],p)[1]
@show ∂k_p_err = abs.(∂k_p_RAD .- ∂k_p_FD) ./ abs.(∂k_p_FD)

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg2(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg2(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD

@show ∂sm4k_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->summag4(solve_k(ms,0.7,rwg(x))[2]),p)[1]
@show ∂sm4k_p_err = abs.(∂sm4k_p_RAD .- ∂sm4k_p_FD) ./ ∂sm4k_p_FD




##
neff1,ng1 = solve_n(ms,om,rwg(p))
neff2,ng2 = solve_n(ms,0.7,rwg2(p))

@show ∂n_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg2(p))[1],0.7)[1]
@show ∂n_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,x,rwg2(p))[1],0.7)[1]
@show ∂n_om_err = abs(∂n_om_RAD - ∂n_om_FD) / abs(∂n_om_FD)

@show ∂ng_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg2(p))[2],0.7)[1]
@show ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,x,rwg2(p))[2],0.7)[1]
@show ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)

@show ∂ng_om_RAD = Zygote.gradient(x->solve_n(ms,x,rwg(p))[2],0.7)[1]
@show ∂ng_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,x,rwg(p))[2],0.7)[1]
@show ∂ng_om_err = abs( ∂ng_om_RAD -  ∂ng_om_FD) /  abs(∂ng_om_FD)

@show ∂n_p_RAD =  Zygote.gradient(x->solve_n(ms,0.7,rwg2(x))[1],p)[1]
@show ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg2(x))[1],p)[1]
@show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

@show ∂n_p_RAD =  Zygote.gradient(x->solve_n(ms,0.7,rwg(x))[1],p)[1]
@show ∂n_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_n(ms,0.7,rwg(x))[1],p)[1]
@show ∂n_p_err = abs.(∂n_p_RAD .- ∂n_p_FD) ./ abs.(∂n_p_FD)

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.7,rwg2(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,0.7,rwg2(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD

@show ∂ng_p_RAD = Zygote.gradient(x->solve_n(ms,0.7,rwg(x))[2],p)[1]
@show ∂ng_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->solve_n(ms,0.7,rwg(x))[2],p)[1]
@show ∂ng_p_err = abs.(∂ng_p_RAD .- ∂ng_p_FD) ./ ∂ng_p_FD


f1(om,pp) = sum(sum(εₛ⁻¹(om,rwg(pp);ms)))
f1(0.7,p)
@show ∂f1_om_RAD = Zygote.gradient(x->f1(x,p),0.7)[1]
@show ∂f1_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f1(x,p),0.7)[1]
@show ∂f1_om_err = abs( ∂f1_om_RAD -  ∂f1_om_FD) /  abs(∂f1_om_FD)

@show ∂f1_p_RAD =  Zygote.gradient(x->f1(0.7,x),p)[1]
@show ∂f1_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->f1(0.7,x),p)[1]
@show ∂f1_p_err = abs.(∂f1_p_RAD .- ∂f1_p_FD) ./ abs.(∂f1_p_FD)

using Zygote: dropgrad
function f2(om,pp)
    ε⁻¹ = εₛ⁻¹(om,rwg(pp);ms)
    k, H⃗ = solve_k(ms,om,ε⁻¹)
    (mag,m⃗,n⃗) = mag_m_n(k,dropgrad(ms.M̂.g⃗))
    om / H_Mₖ_H(H⃗[:,1],ε⁻¹,real(mag),real(reinterpret(reshape,Float64,m⃗)),real(reinterpret(reshape,Float64,n⃗)))
end
f2(0.7,p)
@show ∂f2_om_RAD = Zygote.gradient(x->f2(x,p),0.7)[1]
@show ∂f2_om_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f2(x,p),0.7)[1]
@show ∂f2_om_err = abs( ∂f2_om_RAD -  ∂f2_om_FD) /  abs(∂f2_om_FD)

@show ∂f2_p_RAD =  Zygote.gradient(x->f2(0.7,x),p)[1]
@show ∂f2_p_FD =  FiniteDifferences.grad(central_fdm(5,1),x->f2(0.7,x),p)[1]
@show ∂f2_p_err = abs.(∂f2_p_RAD .- ∂f2_p_FD) ./ abs.(∂f2_p_FD)



∂omsq_p_RAD = Zygote.gradient(x->solve_k(ms,0.7,rwg2(x))[1],p)




∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_ω²(ms,1.45,rwg2(x))[1],p)










∂omsq_p_RAD = Zygote.gradient(x->solve_k(ms,0.65,rwg2(x))[1],p)




∂omsq_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->solve_k(ms,0.65,rwg2(x))[1],p)





∂smm_p_RAD = Zygote.gradient(x->summag4(solve_k(ms,0.65,rwg2(x))[2]),p)[1]




∂smm_p_FD =  FiniteDifferences.grad(central_fdm(3,1),x->summag4(solve_k(ms,0.65,rwg2(x))[2]),p)



##
println("### εₛ: ")
println("## primal: ")
esm = εₛ(0.8,geom,gr)
@btime εₛ(0.8,$geom,$gr) # 2.352 ms (66436 allocations: 7.75 MiB)
println("## gradients: ")
println("# Zygote: ")
@show Zygote.gradient(x->sum(sum(εₛ(x,rwg(p),gr))),0.9)[1]
@show Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]
@btime Zygote.gradient(x->sum(sum(εₛ(x,rwg($p),$gr))),0.9)[1]
@btime Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),$p)[1]
println("# ForwardDiff: ")
@show ForwardDiff.derivative(x->sum(sum(εₛ(x,rwg(p),gr))),0.9)
@show ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)
@btime ForwardDiff.derivative(x->sum(sum(εₛ(x,rwg($p),$gr))),0.9)
@btime ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),$p)
println("# ForwardDiff over Zygote (2nd order): ")
@show ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ(x,rwg(p),gr))),y)[1],0.8)
@show ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),y)[1],p)
@btime ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ(x,rwg($p),$gr))),y)[1],0.8)
@btime ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),$gr))),y)[1],$p)


println("### εₛ⁻¹: ")
println("## primal: ")
eism = εₛ⁻¹(0.8,geom,gr)
@btime εₛ⁻¹(0.8,$geom,$gr) # 2.439 ms (66439 allocations: 7.75 MiB)
println("## gradients: ")
println("# Zygote: ")
@show Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),0.9)[1]
@show Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),p)[1]
@btime Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),0.9)[1]
@btime Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),$p)[1]
println("# ForwardDiff: ")
@show ForwardDiff.derivative(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),0.9)
@show ForwardDiff.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),p)
@btime ForwardDiff.derivative(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),0.9)
@btime ForwardDiff.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),$p)
println("# ForwardDiff over Zygote (2nd order): ")
@show ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg(p),gr))),y)[1],0.8)
@show ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),gr))),y)[1],p)
@btime ForwardDiff.derivative(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(x,rwg($p),$gr))),y)[1],0.8)
@btime ForwardDiff.gradient(y->Zygote.gradient(x->sum(sum(εₛ⁻¹(0.8,rwg(x),$gr))),y)[1],$p)

SMatrix
using ChainRulesCore: NO_FIELDS
ChainRulesCore.rrule(T::Type{<:SMatrix}, x::AbstractMatrix) = ( T(x), dv -> (NO_FIELDS, dv) )
ChainRulesCore.rrule(T::Type{<:SMatrix}, xs::Number...) = ( T(xs...), dv -> (NO_FIELDS, dv...) )

@Zygote.adjoint (T::Type{<:SMatrix})(xs::Number...) = T(xs...), dv -> (nothing, dv...)
@Zygote.adjoint (T::Type{<:SMatrix})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)
@Zygote.adjoint (T::Type{SMatrix{2,2,Float64,4}})(x::AbstractMatrix) = T(x), dv -> (nothing, dv)

ChainRules.refresh_rules()
Zygote.refresh()
Zygote.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]
Zygote.hessian(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)[1]

##



Zygote.gradient(x->Zygote.forwarddiff(y->sum(sum(εₛ(y...))),[0.8,rwg(x),gr]),p)
Zygote.forwarddiff(y->sum(sum(εₛ(y...))),[0.8,rwg(p),gr])
Zygote.gradient(x->sum(sum(εₛ(0.8,Zygote.forwarddiff(rwg,x),gr))),p)
ForwardDiff.gradient(x->sum(sum(εₛ(0.8,rwg(x),gr))),p)

f1(lm,p) = εₛ(lm,rwg(p),gr)
f1(0.8,p)
Zygote.gradient(x->sum(sum(f1(0.8,x))),p)

shapes1 = rwg3(p)
geom2 = rwg2(p)
convert.(Material,getfield.(shapes1,:data))
mats0 = getfield.(shapes1,:data)
similar(mats0,Material)
Material(3.5)
Material.(getfield.(shapes1,:data))

import Base.convert
convert(::Type{Material}, x) = Material(x)
materials(rwg(p))
materials2(shapes::Vector{S}) where S<:Shape{N,N²,D,T} where {N,N²,D<:Material,T} = unique!(getfield.(shapes,:data))
materials2(shapes1)
materials(shapes1)
rwg(p)
eltype(shapes1)<:Shape{N,N²,D,T} where {N,N²,D<:Material,T}
e1 = ε_tensor(3.5)
Material(e1)
##
εs_sym = getfield.(materials(shapes2),:ε)
ε_exprs = build_function.(getfield.(materials(shapes2),:ε),λ)
εs = [ eval(εe[1]) for εe in ε_exprs ]
εs! = [ eval(εe[2]) for εe in ε_exprs ]

εs[1](0.8)

struct Geometry3{N}
	shapes::Vector{Shape{N}}
	# materials::Vector{Material}
end
Geometry3(s::Vector{S}) where S<:Shape{N} where N = Geometry3{N}(s)

shapes1 =
Geometry3(shapes1)


mats = materials(shapes2)
sinds2minds = map(s->findfirst(m->isequal(s.data,m), mats),shapes2)

csinds = corner_sinds(shapes2,ms.M̂.xyz,ms.M̂.xyzc)
sinds_pr = proc_sinds(csinds)
vxl_min = @view ms.M̂.xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
vxl_max = @view ms.M̂.xyzc[min(2,end):end,min(2,end):end,min(2,end):end]



sr1 = S_rvol(ms.M̂.corner_sinds_proc,ms.M̂.xyz,vxl_min,vxl_max,shapes)
@btime S_rvol($ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$vxl_min,$vxl_max,$shapes)

sr2 = S_rvol(ms.M̂.corner_sinds_proc,ms.M̂.xyz,vxl_min,vxl_max,shapes2)
@btime S_rvol($ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$vxl_min,$vxl_max,$shapes2)

corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc)

S_rvol(shapes;ms)
@btime S_rvol($shapes;ms=$ms)
@btime S_rvol(shapes2;ms)

const εᵥ = SMatrix{3,3}(1.,0.,0.,0.,1.,0.,0.,0.,1.)
fεs = map(m->fε(m)[1],mats)
λs = 0.5:0.1:1.6
ωs = 1 ./ λs
εs = [vcat([SMatrix{3,3}(fep(lm)) for fep in fεs],[εᵥ,]) for lm in λs]
minds= matinds(shapes2)

epsm = εₛ(εs[1],ms.M̂.corner_sinds_proc,minds,sr1)
@btime εₛ($εs[1],$ms.M̂.corner_sinds_proc,$minds,$sr1)

εₛ11 = [ee[1,1] for ee in epsm][:,:,1]
εₛ22 = [ee[2,2] for ee in epsm][:,:,1]
εₛ12 = [ee[1,2] for ee in epsm][:,:,1]



geom = Geometry(shapes2)

##
# check that materials/shapes lists and cross-reference index lists work by adding a few shapes
bx1 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[0. , 0.1],            	# c: center
				[2.8, 0.4 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				MgO_LiNbO₃,					        # data: any type, data associated with box shape
			)
bx2 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[-0.5 , 0.4],            	# c: center
				[0.8, 0.2 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				SiO₂,					        # data: any type, data associated with box shape
			)
bx3 = GeometryPrimitives.Box(					                # Instantiate N-D box, here N=2 (rectangle)
				[0.5 , 0.4],            	# c: center
				[0.2, 0.2 ],	# r: "radii" (half span of each axis)
				SMatrix{2,2}(1.,0.,0.,1.),	    		        # axes: box axes
				Si₃N₄,					        # data: any type, data associated with box shape
			)
shapes3 = vcat(shapes2, [bx1,bx2,bx3])
mats3 = materials(shapes3)
sinds2minds3 = map(s->findfirst(m->isequal(s.data,m), mats3),shapes3)


##
struct Material{T}
	ε::SMatrix{3,3,T,9}
end
n(mat::Material) = sqrt.(diag(mat.ε))
n(mat::Material,axind::Int) = sqrt(mat.ε[axind,axind])
ng(mat::Material) = ng_sym.(n.(mat))
ng(mat::Material,axind::Int) = ng_sym(n(mat,axind))
gvd(mat::Material) = gvd_sym.(n.(mat))
gvd(mat::Material,axind::Int) = gvd_sym(n(mat,axind))

using ModelingToolkit
pₑ_MgO_LiNbO₃ = (
    a₁ = 5.756,
    a₂ = 0.0983,
    a₃ = 0.202,
    a₄ = 189.32,
    a₅ = 12.52,
    a₆ = 1.32e-2,
    b₁ = 2.86e-6,
    b₂ = 4.7e-8,
    b₃ = 6.113e-8,
    b₄ = 1.516e-4,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
pₒ_MgO_LiNbO₃ = (
    a₁ = 5.653,
    a₂ = 0.1185,
    a₃ = 0.2091,
    a₄ = 89.61,
    a₅ = 10.85,
    a₆ = 1.97e-2,
    b₁ = 7.941e-7,
    b₂ = 3.134e-8,
    b₃ = -4.641e-9,
    b₄ = -2.188e-6,
    T₀ = 24.5,      # reference temperature in [Deg C]
)
function n²_MgO_LiNbO₃_sym(λ, T; a₁, a₂, a₃, a₄, a₅, a₆, b₁, b₂, b₃, b₄, T₀)
    f = (T - T₀) * (T + T₀ + 2*273.16)  # so-called 'temperature dependent parameter'
    λ² = λ^2
    a₁ + b₁*f + (a₂ + b₂*f) / (λ² - (a₃ + b₃*f)^2) + (a₄ + b₄*f) / (λ² - a₅^2) - a₆*λ²
end
@variables λ, T
nₑ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₑ_MgO_LiNbO₃...)
nₑ²_MgO_LiNbO₃_sym = substitute(nₑ²_MgO_LiNbO₃_λT_sym,[T=>pₑ_MgO_LiNbO₃.T₀])
nₒ²_MgO_LiNbO₃_λT_sym = n²_MgO_LiNbO₃_sym(λ, T; pₒ_MgO_LiNbO₃...)
nₒ²_MgO_LiNbO₃_sym = substitute(nₒ²_MgO_LiNbO₃_λT_sym,[T=>pₒ_MgO_LiNbO₃.T₀])
ε_MgO_LiNbO₃_λT_sym = Diagonal( [ nₑ²_MgO_LiNbO₃_λT_sym, nₒ²_MgO_LiNbO₃_λT_sym, nₒ²_MgO_LiNbO₃_λT_sym ] )
ε_MgO_LiNbO₃_sym = Diagonal( [ nₑ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym, nₒ²_MgO_LiNbO₃_sym ] )
LN = Material(SMatrix{3,3}(ε_MgO_LiNbO₃_sym))

function materials(shapes::Vector{Shape{N,N²,D,T}}) where {N,N²,D,T}
	unique!(getfield.(shapes,:data))
end

materials2(shapes) = unique!(getfield.(shapes,:data))

struct Geometry3{N,N²,D,T}
	shapes::Vector{Shape{N,N²,D,T}}
end

wg1 = Geometry3(rwg(p))
shapes3 = vcat(shapes,shapes,shapes,shapes,shapes)
wg3 = Geometry3(shapes3)
##
xy = [ SVector(ms.M̂.x[i],ms.M̂.y[j]) for i=1:Ny,j=1:Nx ]
xyc = [SVector{2}(ms.M̂.xc[i],ms.M̂.yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]

update_corner_sinds!(ms.M̂.corner_sinds,shapes,xy,xyc)
@btime update_corner_sinds!($ms.M̂.corner_sinds,$shapes,$xy,$xyc)

update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc)
@btime update_corner_sinds!($ms.M̂.corner_sinds,$shapes,$ms.M̂.xyz,$ms.M̂.xyzc)

proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc)
@btime proc_corner_sinds!($ms.M̂.corner_sinds,$ms.M̂.corner_sinds_proc)


function τ_trans(ε::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/ε[1,1],      ε[2,1]/ε[1,1],                  ε[3,1]/ε[1,1],
        ε[1,2]/ε[1,1],  ε[2,2] - ε[2,1]*ε[1,2]/ε[1,1],  ε[3,2] - ε[3,1]*ε[1,2]/ε[1,1],
        ε[1,3]/ε[1,1],  ε[2,3] - ε[2,1]*ε[1,3]/ε[1,1],  ε[3,3] - ε[3,1]*ε[1,3]/ε[1,1]
    )
end

function τ⁻¹_trans(τ::AbstractMatrix{T}) where T<:Real
    return @inbounds SMatrix{3,3,T,9}(
        -1/τ[1,1],          -τ[2,1]/τ[1,1],                 -τ[3,1]/τ[1,1],
        -τ[1,2]/τ[1,1],     τ[2,2] - τ[2,1]*τ[1,2]/τ[1,1],  τ[3,2] - τ[3,1]*τ[1,2]/τ[1,1],
        -τ[1,3]/τ[1,1],     τ[2,3] - τ[2,1]*τ[1,3]/τ[1,1],  τ[3,3]- τ[3,1]*τ[1,3]/τ[1,1]
    )
end

function avg_param(ε_fg, ε_bg, n12, rvol1)
	n = n12 / norm(n12)
	# n = normalize(n12) #n12 / norm(n12) #sqrt(sum2(abs2,n12))
    # Pick a vector that is not along n.
    h = any(iszero.(n)) ? n × normalize(iszero.(n)) :  n × SVector(1., 0. , 0.)
	v = n × h
    # Create a local Cartesian coordinate system.
    S = [n h v]  # unitary
    τ1 = τ_trans(transpose(S) * ε_fg * S)  # express param1 in S coordinates, and apply τ transform
    τ2 = τ_trans(transpose(S) * ε_bg * S)  # express param2 in S coordinates, and apply τ transform
    τavg = τ1 .* rvol1 + τ2 .* (1-rvol1)  # volume-weighted average
    return SMatrix{3,3}(S * τ⁻¹_trans(τavg) * transpose(S))  # apply τ⁻¹ and transform back to global coordinates
end

# alleq(itr) = length(itr)==0 || all( ==(itr[1]), itr)
get_ε(shapes,ind) = ind>lastindex(shapes) ? SMatrix{3,3}(1.,0.,0.,0.,1.,0.,0.,0.,1.) : shapes[ind].data
V3(v) = isequal(length(v),3) ? v : vcat(v,zeros(3-length(v)))

function n_rvol(shape,xyz,vxl_min,vxl_max)
	r₀,n⃗ = surfpt_nearby(xyz, shape)
	rvol = volfrac((vxl_min,vxl_max),n⃗,r₀)
	return V3(n⃗),rvol
end

function _smooth(shapes,sinds_proc,xyz,vxl_min,vxl_max)
	iszero(sinds_proc[2]) && return get_ε(shapes,sinds_proc[1])
	iszero(sinds_proc[3]) && return avg_param(	shapes[sinds_proc[1]].data,
												get_ε(shapes,sinds_proc[2]),
												n_rvol(shapes[sinds_proc[1]],xyz,vxl_min,vxl_max)...
												)
	return mapreduce(i->get_ε(shapes,i),+,sinds_proc) / 8
end

function smooth(shapes,sinds_proc,xyz,xyzc)
	vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		_smooth(s,sp,x,vn,vp)
	end
	map(f,sinds_proc,xyz,vxl_min,vxl_max)
end

# smooth(shapes,ms::ModeSolver) = smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)

function smooth(shapes;ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
	HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}(
		reshape(
			reinterpret(
				reshape,
				Float64,
				smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
				),
			(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
		)
	)
end

function smoothinv(shapes,sinds_proc,xyz,xyzc)
	vxl_min = @view xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		inv(_smooth(s,sp,x,vn,vp))
	end
	map(f,sinds_proc,xyz,vxl_min,vxl_max)
end

function smoothinv(shapes;ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
	HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}(
		reshape(
			reinterpret(
				reshape,
				Float64,
				smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
				),
			(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
		)
	)
end

using StaticArrays: Dynamic
function epsi(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	vxl_min = @view ms.M̂.xyzc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
	vxl_max = @view ms.M̂.xyzc[min(2,end):end,min(2,end):end,min(2,end):end]
	f(sp,x,vn,vp) = let s=shapes
		inv(_smooth(s,sp,x,vn,vp))
	end
	eibuf = Zygote.Buffer(bounds(shapes[1])[1],3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz)
	# eibuf = Buffer(bounds(shapes[1])[1],3,3,Nx,Ny,Nz)
    for ix=1:ms.M̂.Nx,iy=1:ms.M̂.Ny,iz=1:ms.M̂.Nz
		# eps = εₛ(shapes,Zygote.dropgrad(tree),Zygote.dropgrad(g.x[i]),Zygote.dropgrad(g.y[j]),Zygote.dropgrad(g.δx),Zygote.dropgrad(g.δy))
		# eps = εₛ(shapes,x[i],y[j];tree,δx,δy)
		# epsi = inv(eps) # inv( (eps' + eps) / 2) # Hermitian(inv(eps))  # inv(Hermitian(eps)) #   # inv(eps)
        # eibuf[:,:,i,j,kk] = epsi #(epsi' + epsi) / 2
		eibuf[:,:,ix,iy,iz] = f(ms.M̂.corner_sinds_proc[ix,iy,iz],ms.M̂.xyz[ix,iy,iz],vxl_min[ix,iy,iz],vxl_max[ix,iy,iz])
    end
    # return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()},T,5,5,Array{T,5}}( real(copy(eibuf)) )
	return HybridArray{Tuple{3,3,Dynamic(),Dynamic(),Dynamic()}}( real(copy(eibuf)) )
end

function epsi2(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
end

function epsi3(shapes::Vector{<:Shape};ms::ModeSolver)
	Zygote.@ignore(update_corner_sinds!(ms.M̂.corner_sinds,shapes,ms.M̂.xyz,ms.M̂.xyzc))
	Zygote.@ignore(proc_corner_sinds!(ms.M̂.corner_sinds,ms.M̂.corner_sinds_proc))
	reshape(
		reinterpret(
			reshape,
			Float64,
			smoothinv(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc),
			),
		(3,3,ms.M̂.Nx,ms.M̂.Ny,ms.M̂.Nz),
	)
end

##
epsi(shapes;ms)
@btime epsi($shapes;ms=$ms) #64.106 ms (613646 allocations: 27.62 MiB)

epsi2(shapes;ms)
@btime epsi2($shapes;ms=$ms) # 2.991 ms (105718 allocations: 6.86 MiB)

epsi3(shapes;ms)
@btime epsi3($shapes;ms=$ms) # 3.246 ms (105721 allocations: 6.86 MiB)

smooth(shapes,ms.M̂.corner_sinds_proc,xy,xyc)
@btime smooth($shapes,$ms.M̂.corner_sinds_proc,$xy,$xyc)

smooth(shapes,ms.M̂.corner_sinds_proc,ms.M̂.xyz,ms.M̂.xyzc)
@btime smooth($shapes,$ms.M̂.corner_sinds_proc,$ms.M̂.xyz,$ms.M̂.xyzc)

smooth(shapes,ms)
@btime smooth($shapes,$ms)

es = smooth(shapes,ms)

# smooth2(shapes,ms)
# @btime smooth2($shapes,$ms)

# Compare with old smoothing function
tree = tree(shapes)
es_old = [SMatrix{3,3}(εₛ(shapes,ms.M̂.x[xind],ms.M̂.y[yind];tree,δx=ms.M̂.δx,δy=ms.M̂.δy)) for xind=1:Nx,yind=1:Ny]
@assert all(es_old .≈ es[:,:,1])
@btime [SMatrix{3,3}(εₛ($shapes,$ms.M̂.x[xind],$ms.M̂.y[yind];tree,δx=ms.M̂.δx,δy=ms.M̂.δy)) for xind=1:Nx,yind=1:Ny]
# 296.386 ms (1724616 allocations: 75.20 MiB)

using ChainRules, Zygote, ForwardDiff, FiniteDifferences
f1(x) = sum(sum(smooth(rwg(x);ms)))/(128*128)
f1(p)
f2(x) = sum(sum(smoothinv(rwg(x);ms)))/(128*128)
f2(p)
Zygote.gradient(f1,p)[1]
ForwardDiff.gradient(f1,p)
FiniteDifferences.grad(central_fdm(3,1),f1,p)

println("######  btimes for f1, using regular map:")
println("f1:")
@btime f1($p)
println("FowardDiff:")
@btime ForwardDiff.gradient($f1,$p)
println("Zygote:")
@btime FiniteDifferences.grad(central_fdm(3,1),$f1,$p)
println("FiniteDifferences:")
@btime Zygote.gradient($f1,$p)[1]

println("######  btimes for f2, using pmap:")
println("f2:")
@btime f2($p)
println("FowardDiff:")
@btime ForwardDiff.gradient($f2,$p)
println("Zygote:")
@btime FiniteDifferences.grad(central_fdm(3,1),$f2,$p)
println("FiniteDifferences:")
@btime Zygote.gradient($f2,$p)[1]
# ######  btimes for f1, using regular map:
# FowardDiff:
# 8.280 ms (57116 allocations: 14.59 MiB)
# Zygote:
# 67.081 ms (1147885 allocations: 147.68 MiB)
# FiniteDifferences:
# 678.245 ms (3484021 allocations: 171.39 MiB)
# ######  btimes for f2, using pmap:
# FowardDiff:
# 8.781 ms (57116 allocations: 14.59 MiB)
# Zygote:
# 69.119 ms (1147885 allocations: 147.68 MiB)
# FiniteDifferences:
# 630.052 ms (3484021 allocations: 171.39 MiB)

##
vxl_min2 = @view xyc[1:max((end-1),1),1:max((end-1),1),1:max((end-1),1)]
vxl_max2 = @view xyc[min(2,end):end,min(2,end):end,min(2,end):end]
fsm(sp,x,vn,vp) = let s=shapes
	_smooth(s,sp,x,vn,vp)
end
##
# corner condition where simple averaging should occur: CartesianIndex(44, 54)

I = CartesianIndex(44,54,1)
_smooth(shapes,ms.M̂.corner_sinds_proc[I],xy[I],vxl_min2[I],vxl_max2[I])
_smooth(shapes,ms.M̂.corner_sinds_proc[I],ms.M̂.xyz[I],vxl_min2[I],vxl_max2[I])

##

function avg_param2(xy,sinds)
        r₀,nout = surfpt_nearby(xy, shapes[sinds[0]])

end

shapes = rwg(p)
tree = KDTree(shapes)
n_shapes = length(shapes)

# gridpoint positions
x = ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.
y = ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.
xy = [ SVector(x[i],y[j]) for i=1:Ny,j=1:Nx ]
# corner positions
xc = ( ( Δx / Nx ) .* (0:Nx) ) .- ( Δx/2. * ( 1 + 1. / Nx ) )
yc = ( ( Δy / Ny ) .* (0:Ny) ) .- ( Δy/2. * ( 1 + 1. / Ny ) )
xyc = [SVector{2}(xc[i],yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]
sc = Array{Int}(undef,size(xyc))
sc_ext = Array{NTuple{4,Int}}(undef,size(xy))
sc .= [(a = findfirst(isequal(findfirst(SVector(xyc[i,j]),tree)),shapes); isnothing(a) ? (n_shapes+1) : a ) for i=1:(Nx+1),j=(1:Ny+1)]
sc_ext .= [ (unq = unique!( [sc[i,j], sc[1+1,j], sc[i+1,j+1], sc[i,j+1]] ); n_unq=length(unq); n_unq==1 ? (unq[1],0,0,0) : ( n_unq==2 ?  (minimum(unq),maximum(unq),0,0)  : ( sc[i,j],  sc[i+1,j],  sc[i+1,j+1],  sc[i,j+1] ) ) )  for i=1:Nx,j=1:Ny ]

sc

128 * 128



##
# gridpoint positions
x = ( ( Δx / Nx ) .* (0:(Nx-1))) .- Δx/2.
y = ( ( Δy / Ny ) .* (0:(Ny-1))) .- Δy/2.
z = ( ( Δz / Nz ) .* (0:(Nz-1))) .- Δz/2.
xy = [ SVector(x[i],y[j]) for i=1:Ny,j=1:Nx ]
xyz = [ SVector{3}(x[i],y[j],z[k]) for i=1:Ny,j=1:Nx,k=1:Nz ]
# corner positions
xc = ( ( Δx / Nx ) .* (0:Nx) ) .- ( Δx/2. * ( 1 + 1. / Nx ) )
yc = ( ( Δy / Ny ) .* (0:Ny) ) .- ( Δy/2. * ( 1 + 1. / Ny ) )
zc = ( ( Δz / Nz ) .* (0:Nz) ) .- ( Δz/2. * ( 1 + 1. / Nz ) )
xyc = [SVector{2}(xc[i],yc[j]) for i=1:(Nx+1),j=1:(Ny+1)]
xyzc = [SVector{3}(xc[i],yc[j],zc[k]) for i=1:(Nx+1),j=1:(Ny+1),k=1:(Nz+1)]
# arrays for shape index data

corner_sinds2 = zeros(Int, Nx+1,Ny+1)
corner_sinds_proc2 = fill((0,0,0,0), Nx,Ny)
corner_sinds3 = zeros(Int, Nx+1,Ny+1,Nz+1)
corner_sinds_proc3 = fill((0,0,0,0,0,0,0,0), Nx,Ny,Nz)

# update_corner_sinds!(corner_sinds2,corner_sinds_proc2,shapes,xy,xyc)
# @btime update_corner_sinds!($corner_sinds2,$corner_sinds_proc2,$shapes,$xy,$xyc)

update_corner_sinds4!(corner_sinds2,corner_sinds_proc2,shapes,xy,xyc)
@btime update_corner_sinds4!($corner_sinds2,$corner_sinds_proc2,$shapes,$xy,$xyc)

##
function update_corner_sinds!(corner_sinds,corner_sinds_proc,shapes::AbstractVector{<:GeometryPrimitives.Shape{2}},xyz,xyzc)
	n_shapes = length(shapes)
	tree = KDTree(shapes)
	a = 0
	unq = [0,0]
	for I ∈ eachindex(xyzc)
		a = findfirst(isequal(findfirst(xyzc[I],tree)),shapes)
		corner_sinds[I] = isnothing(a) ? (n_shapes+1) : a
	end
	for I ∈ CartesianIndices(xyz)
		unq = [	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)],
		  		]
		# unq = unique!( unq )
		unique!( unq )
		a = length(unq)
		corner_sinds_proc[I] = a==1 ? (unq[1],0,0,0) :
			( a==2 ?  (minimum(unq),maximum(unq),0,0)  :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0)],
					corner_sinds[I+CartesianIndex(0,1)],
					corner_sinds[I+CartesianIndex(1,1)],
				)
		)
	end
end

in3(x::SVector{2,<:Real}, s::GeometryPrimitives.Polygon) = all(sum(s.n .* (x' .- s.v), dims=Val(2)) .≤ 0)

function in3(x::SVector{N,<:Real}, b::GeometryPrimitives.Box{N}) where {N}
    d = b.p * (x - b.c)
    for i = 1:N
        abs(d[i]) > b.r[i] && return false  # boundary is considered inside
    end
    return true
end

function f_cinds2(shapes::Vector{S}) where {S<:Shape{N}} where N #::Int
	function f_cinds2_inner(p)
		let s=shapes # pairs(shapes)
			# x-> something(ff4(x,s),lp1) #::Int
			@inbounds for (i, a) in pairs(s)
				in3(p,a) && return i
			end
			return lastindex(s)+1
		end
	end
end

function update_corner_sinds4!(corner_sinds,corner_sinds_proc,shapes,xyz,xyzc)
	# a = 0
	unq = [0,0]
	function f_cinds(p::SVector{N,T}) where {N,T} #::Int
		let s=shapes # pairs(shapes)
			# x-> something(ff4(x,s),lp1) #::Int
			for (i, a) in pairs(s)
		        in3(p,a) && return i
				# true && return i
		    end
		    return lastindex(s)+1
		end
	end
	# corner_sinds .= f_cinds.(xyzc)
	# map!(corner_sinds,xyzc) do p
	# 	let s=shapes
	# 		for (i, a) in pairs(s)
	# 			in3(p,a) && return i
	# 		end
	# 		return lastindex(s)+1
	# 	end
	# end
	@inbounds for I in eachindex(xyzc)
		corner_sinds[I] = f_cinds(xyzc[I]) #::Int
	end
	for I ∈ CartesianIndices(xyz)
	 	unq = [		corner_sinds[I],
								corner_sinds[I+CartesianIndex(1,0)],
								corner_sinds[I+CartesianIndex(0,1)],
								corner_sinds[I+CartesianIndex(1,1)]
			  				]
		# unq = unique!( unq )
		unique!( unq )
		a = length(unq)
		println("f0")
		corner_sinds_proc[I] = isone(a) ? (unq[1],0,0,0) :
			( a===2 ?  (minimum(unq),maximum(unq),0,0)  :
				( corner_sinds[I],
							corner_sinds[I+CartesianIndex(1,0)],
							corner_sinds[I+CartesianIndex(0,1)],
							corner_sinds[I+CartesianIndex(1,1)]
						)
		)
	end
end




function update_corner_sinds!(corner_sinds,corner_sinds_proc,shapes::AbstractVector{<:GeometryPrimitives.Shape{3}},xyz,xyzc)
	n_shapes = length(shapes)
	tree = KDTree(shapes)
	a = 0
	unq = [0,0]
	for I ∈ eachindex(xyzc)
		a = findfirst(isequal(findfirst(xyzc[I],tree)),shapes)
		corner_sinds[I] = isnothing(a) ? (n_shapes+1) : a
	end
	for I ∈ eachindex(xyz)
		unq .= [	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)],
		  		]
		unq = unique!( unq )
		a = length(unq)
		corner_sinds_proc[I] = a==1 ? (unq[1],0,0,0,0,0,0,0) :
			( a==2 ?  (minimum(unq),maximum(unq),0,0,0,0,0,0)  :
				( 	corner_sinds[I],
					corner_sinds[I+CartesianIndex(1,0,0)],
					corner_sinds[I+CartesianIndex(0,1,0)],
					corner_sinds[I+CartesianIndex(1,1,0)],
					corner_sinds[I+CartesianIndex(0,0,1)],
					corner_sinds[I+CartesianIndex(1,0,1)],
					corner_sinds[I+CartesianIndex(0,1,1)],
					corner_sinds[I+CartesianIndex(1,1,1)],
				)
		)
	end
end

function ff2(p::SVector{N}, s::Vector{S}) where {N,S<:Shape{N,N²,D,T}} where {N²,D,T<:Real}
    # for i in eachindex(s)
    #     b::Tuple{SVector{2,T}, SVector{2,T}} = bounds(s[i])
    #     # if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
	# 	if in(p, s[i])  # check if p is within bounding box is faster
    #         return s[i]
    #     end
    # end
	for ss in s
        # b::Tuple{SVector{2,T}, SVector{2,T}} = bounds(ss)
        # if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
		if in(p,ss)  # check if p is within bounding box is faster
            return ss
        end
    end
	# return s[1]
	return nothing
end

function ff3(p::SVector{N}, s::Vector{S}) where {N,S<:Shape{N,N²,D,T}} where {N²,D,T<:Real}
	pin = let p = p
		x->in(p,x)
	end
	findfirst(pin,s)
	# findfirst(x->in(p,x),s)
end

ff4(p,s) = let p=p
	findfirst(x->in(p,x),s)
end



ff5(p::SVector{2,T},s) where T<:Real = findfirst(x->in(p,x),s)

function ff6(p,s)
	let p=p
		y = findfirst(x->in(p,x),s)
		!isnothing(y) ? y : length(s)+1
	end
end

function ff2(p::SVector{N}, kd::KDTree{N}) where {N}
    if isempty(kd.s)
        if p[kd.ix] ≤ kd.x
            return ff2(p, kd.left)
        else
            return ff2(p, kd.right)
        end
    else
        return ff2(p, kd.s)
    end
end

function ff2(p::SVector{N}, s::Vector{S}, sbg::S) where {N,S<:Shape{N}}
    @inbounds for i in eachindex(s)
        @inbounds b::Tuple{SVector{2}, SVector{2}} = bounds(s[i])
        @inbounds if all(b[1] .< p .< b[2]) && p ∈ s[i]  # check if p is within bounding box is faster
            @inbounds return s[i]
        end
    end
    return sbg
end

function ff2(p::SVector{N}, kd::KDTree{N}, sbg::Shape{N}) where {N}
    @inbounds if isempty(kd.s)
        @inbounds if p[kd.ix] ≤ kd.x
            return ff2(p, kd.left, sbg)
        else
            return ff2(p, kd.right, sbg)
        end
    else
        return ff2(p, kd.s, sbg)
    end
end
