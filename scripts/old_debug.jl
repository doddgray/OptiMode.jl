using Revise
using OptiMode
using LinearAlgebra, LinearMaps, IterativeSolvers, FFTW, ChainRulesCore, ChainRules, Zygote, OptiMode, BenchmarkTools
include("mpb_example.jl")
ε⁻¹ = [ε⁻¹_mpb[i,j,k][a,b] for a=1:3,b=1:3,i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz];
ω = ω_mpb
H,k = solve_k(ω_mpb,ε⁻¹)
Nz = 1
gx = collect(fftfreq(Nx,Nx/6.0))
gy = collect(fftfreq(Ny,Ny/4.0))
gz = collect(fftfreq(Nz,Nz/1.0))
Nx,Ny,Nz = size(ε⁻¹)[end-2:end]

function foo(ω,ε⁻¹) #Δx,Δy)
	H,kz = solve_k(ω, ε⁻¹) # out = real(norm(H) * k)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	# return -ω / real(dot(vec(Ha), vec(zyg_Mₖ(Ha,ε⁻¹,Zygote.hook(ā -> real(ā), kz),gx,gy,gz))))
	return -ω / real(dot(vec(Ha), vec(zyg_Mₖ(Ha,ε⁻¹,kz,gx,gy,gz))))
end

function foo2(ω,ε⁻¹,Δx,Δy,Δz)
	Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
	gx = Zygote.@ignore collect(fftfreq(Nx,Nx/Δx))
	gy = Zygote.@ignore collect(fftfreq(Ny,Ny/Δy))
	gz = Zygote.@ignore collect(fftfreq(Nz,Nz/Δz))
	H,kz = solve_k(ω, ε⁻¹) # out = real(norm(H) * k)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	# return -ω / real(dot(vec(Ha), vec(zyg_Mₖ(Ha,ε⁻¹,Zygote.hook(ā -> real(ā), kz),gx,gy,gz))))
	return -ω / real(dot(vec(Ha), vec(zyg_Mₖ(Ha,ε⁻¹,kz,gx,gy,gz))))
end

########### test to fix broadcasting/allocation slowdown of ng gradient calc

function sum2(op,arr)
    return sum(op,arr)
end

function sum2adj( Δ, op, arr )
    n = length(arr)
    g = x->Δ*Zygote.gradient(op,x)[1]
    return ( nothing, map(g,arr))
end

Zygote.@adjoint function sum2(op,arr)
    return sum2(op,arr),Δ->sum2adj(Δ,op,arr)
end

Zygote.refresh()

function KpG(kz::Real,gx::Real,gy::Real,gz::Real)
	# scale = ds.kpG[i,j,k].mag
	kpg = [-gx; -gy; kz-gz]
	mag = sqrt(sum2(abs2,kpg)) # norm(kpg)
	if mag==0
		n = [0.; 1.; 0.] # SVector(0.,1.,0.)
		m = [0.; 0.; 1.] # SVector(0.,0.,1.)
	else
		if kpg[1]==0. && kpg[2]==0.    # put n in the y direction if k+G is in z
			n = [0.; 1.; 0.] #SVector(0.,1.,0.)
		else                                # otherwise, let n = z x (k+G), normalized
			ntemp = [0.; 0.; 1.] × kpg  #SVector(0.,0.,1.) × kpg
			n = ntemp / sqrt(sum2(abs2,ntemp)) # norm(ntemp) #
		end
	end
	# m = n x (k+G), normalized
	mtemp = n × kpg
	m = mtemp / sqrt(sum2(abs2,mtemp)) #norm(mtemp) # sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
	return kpg, mag, m, n
end

function KpG2(kz::Float64,gx::Float64,gy::Float64,gz::Float64)
	# scale = ds.kpG[i,j,k].mag
	kpg = SVector{3,Float64}(-gx, -gy, kz-gz)
	mag = norm(kpg)
	if mag==0
		n = SVector{3,Float64}(0.,1.,0.)
		m = SVector{3,Float64}(0.,0.,1.)
	else
		if kpg[1]==0. && kpg[2]==0.    # put n in the y direction if k+G is in z
			n = SVector{3,Float64}(0.,1.,0.)
		else                                # otherwise, let n = z x (k+G), normalized
			ntemp = SVector{3,Float64}(0.,0.,1.) × kpg
			n = ntemp / norm(ntemp) #
		end
	end
	# m = n x (k+G), normalized
	mtemp = n × kpg
	m = mtemp / norm(mtemp) # sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
	return kpg, mag, m, n
end

kpg, mag, m, n = KpG(1.5,3.5,6.5,0.0)
KpG2(1.5,3.5,6.5,0.0)
KpG(1.5,3.5,6.5,0.0)[2]
KpG2(1.5,3.5,6.5,0.0)[2]
Zygote.gradient((kz,gx,gy,gz)->KpG(kz,gx,gy,gz)[2],1.5,3.5,6.5,0.0)
Zygote.gradient((kz,gx,gy,gz)->KpG2(kz,gx,gy,gz)[2],1.5,3.5,6.5,0.0)

function ∇KpG2(a,b,c,d)
	Zygote.gradient((kz,gx,gy,gz)->KpG(kz,gx,gy,gz)[2],a,b,c,d)
end

function ∇KpG22(a,b,c,d)
	Zygote.gradient((kz::Float64,gx::Float64,gy::Float64,gz::Float64)->KpG2(kz,gx,gy,gz)[2]::Float64,a,b,c,d)
end


@btime ∇KpG2(1.5,3.5,6.5,0.0)
@btime ∇KpG22(1.5,3.5,6.5,0.0)

function new_zyg_zcross_t2c(H,kpg)
	# [-H[1] * m[2] - Hin[2] * n[2] ; Hin[1] * m[1] + Hin[2] * n[1]; 0.0 ]
	[-H[1] * kpg[3][2] - H[2] * kpg[4][2] ; H[1] * kpg[3][1] + H[2] * kpg[4][1]; 0.0 ]
end

function new_zyg_kcross_c2t(H,kpg)
	# [dot(H,kpg[4])*kpg[2];-dot(H,kpg[3])*kpg[2]]
	SVector{2,ComplexF64}(-dot(H,kpg[4])*kpg[2],dot(H,kpg[3])*kpg[2])
end


# kpgA = reshape([new_zyg_KplusG(kz,ggx,ggy,ggz) for ggx in gx, ggy in gy, ggz in gz], (1,Nx,Ny,Nz))
kpgA = [new_zyg_KplusG(kz,ggx,ggy,ggz) for ggx in gx, ggy in gy, ggz in gz]
HA = reshape(H,(2,Nx,Ny,Nz))
HSA = [SVector{2,ComplexF64}(HA[:,i,j,k]) for i=1:Nx,j=1:Ny,k=1:Nz]


d_HSA = SVector{3,ComplexF64}.([fft([dd[a] for dd in new_zyg_zcross_t2c.(HSA,kpgA)]) for a=1:3]...)
e_HSA = SVector{3,ComplexF64}.([ifft([ee[b] for ee in ε⁻¹_mpb .* d_HSA]) for b=1:3]...)
Mk_HSA = new_zyg_kcross_c2t.(e_HSA,kpgA)

# Mk_HSA = new_zyg_kcross_c2t.(SVector{3,ComplexF64}.([ifft([ee[b] for ee in ε⁻¹_mpb .* SVector{3,ComplexF64}.([fft([dd[a] for dd in new_zyg_zcross_t2c.(HSA,kpgA)]) for a=1:3]...)] ) for b=1:3]...),kpgA)
Mk_HA = reinterpret(ComplexF64,reshape(Mk_HSA,(1,Nx,Ny,Nz)))

# sum(dot.(HSA,new_zyg_kcross_c2t.(SVector{3,ComplexF64}.([ifft([ee[b] for ee in ε⁻¹_mpb .* SVector{3,ComplexF64}.([fft([dd[a] for dd in new_zyg_zcross_t2c.(HSA,kpgA)]) for a=1:3]...)] ) for b=1:3]...),kpgA) ) )


function new_zyg_Mₖ(H,ε⁻¹,kz,gx,gy,gz)
    # Hout = zeros(ComplexF64,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
	Hin = reshape(H,(2,Nx,Ny,Nz))
	H1 = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	#H2 = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = new_zyg_KplusG(kz,gx[i],gy[j],gz[k])
        H1[1,i,j,k] = -Hin[1,i,j,k] * m[2] - Hin[2,i,j,k] * n[2]
        H1[2,i,j,k] =  Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * n[1]
        H1[3,i,j,k] = 0.0
    end
	H11 = copy(H1)
    H2 = fft(H11,(2:4))
	println("$(typeof(H2))")
	H3 = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
        H3[1,i,j,k] =  ε⁻¹[1,1,i,j,k]*H2[1,i,j,k] + ε⁻¹[2,1,i,j,k]*H2[2,i,j,k] + ε⁻¹[3,1,i,j,k]*H2[3,i,j,k]
        H3[2,i,j,k] =  ε⁻¹[1,2,i,j,k]*H2[1,i,j,k] + ε⁻¹[2,2,i,j,k]*H2[2,i,j,k] + ε⁻¹[3,2,i,j,k]*H2[3,i,j,k]
        H3[3,i,j,k] =  ε⁻¹[1,3,i,j,k]*H2[1,i,j,k] + ε⁻¹[2,3,i,j,k]*H2[2,i,j,k] + ε⁻¹[3,3,i,j,k]*H2[3,i,j,k]
    end
	H33 = copy(H3)
    H4 = ifft(H33,(2:4))
	Hout = Zygote.Buffer(Hin,2,Nx,Ny,Nz)
    @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = new_zyg_KplusG(kz,gx[i],gy[j],gz[k])
        scale = -1 / mag
        at1 = H4[1,i,j,k] * m[1] + H4[2,i,j,k] * m[2] + H4[3,i,j,k] * m[3]
        at2 = H4[1,i,j,k] * n[1] + H4[2,i,j,k] * n[2] + H4[3,i,j,k] * n[3]
        Hout[1,i,j,k] =  at2 / mag
        Hout[2,i,j,k] =  -at1 / mag
    end
    return copy(Hout)
end

function foo3(ω,ε⁻¹,Δx,Δy,Δz)
	Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
	gx = Zygote.@ignore collect(fftfreq(Nx,Nx/Δx))
	gy = Zygote.@ignore collect(fftfreq(Ny,Ny/Δy))
	gz = Zygote.@ignore collect(fftfreq(Nz,Nz/Δz))
	H,kz = solve_k(ω, ε⁻¹) # out = real(norm(H) * k)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	# return -ω / real(dot(vec(Ha), vec(zyg_Mₖ(Ha,ε⁻¹,Zygote.hook(ā -> real(ā), kz),gx,gy,gz))))
	return -ω / real(dot(vec(Ha), vec(new_zyg_Mₖ(H,ε⁻¹,kz,gx,gy,gz))))
end


function fooRD(ωA,ε⁻¹) #Δx,Δy)
	ω = ωA[1]
	H,kz = solve_k(ω, ε⁻¹) # out = real(norm(H) * k)
	Ha = reshape(H,(2,Nx,Ny,Nz))
	return -ω / real(dot(vec(Ha), vec(zyg_Mₖ(Ha,ε⁻¹,kz,gx,gy,gz))))
end

foo(ω_mpb, ε⁻¹)
foo2(ω_mpb, ε⁻¹,6.0,4.0,1.0)
foo3(ω_mpb, ε⁻¹,6.0,4.0,1.0)

Zygote.gradient(foo,ω_mpb,ε⁻¹)
Zygote.gradient(foo2,ω_mpb,ε⁻¹,6.0,4.0,1.0)
Zygote.gradient(foo3,ω_mpb,ε⁻¹,6.0,4.0,1.0)
Zygote.gradient(foo2,ω_mpb,ε⁻¹,Δx=6.0,Δy=4.0,Δz=1.0)

fooRD([ω_mpb, ] , ε⁻¹)
Zygote.refresh()
##
ng_primal, foo_pb = Zygote.pullback(foo,ω,ε⁻¹)



H̄ = rand(ComplexF64,size(ds.H⃗))[:,1]
k̄ = rand(Float64)
ΔΩ = (H̄, k̄)

foo_pb(H̄, k̄)  #(ΔΩ)

solve_k_pullback(H̄, k̄, ω_mpb, ε⁻¹_mpb;ds,neigs=1,eigind=1,maxiter=3000,tol=1e-8)


##
function zyg_ng(H,eps,kz,gx,gy,gz)
	return real(dot(vec(H), vec(zyg_Mₖ(H,eps,kz,gx,gy,gz))))
end



zyg_ng(reshape(H,(2,Nx,Ny,Nz)),ε⁻¹,k,fftfreq(Nx,Nx/6.0),fftfreq(Ny,Ny/6.0),fftfreq(Nz,Nz/1.0))



using ReverseDiff
const foo_tape = ReverseDiff.GradientTape(fooRD,([ω_mpb, ],ε⁻¹))

ChainRules.refresh_rules()
Zygote.refresh()


##


epsR = Array{Real,5}(eps);
gx = [g[1] for g in ds.gx]; gy = [g[2] for g in ds.gy]; gz = [g[3] for g in ds.gz];
Hin = copy(reshape(H,(2,ds.Nx,ds.Ny,ds.Nz)));
kz = Real(1.5)
kzA = [kz,]
kzC = ComplexF64(kz)
kzCA = [ComplexF64(kz),]
Nx,Ny,Nz = size(ε⁻¹_mpb)[end-2:end]
g = MaxwellGrid(6.0,4.0,1.0,Nx,Ny,Nz)
k_guess=ω_mpb*sqrt(1/minimum([minimum(eps[a,a,:,:,:]) for a=1:3]))
ds = MaxwellData(k_guess,g)
ε⁻¹_SHM3 = [SHM3(eps[:,:,i,j,kk]) for i=1:Nx,j=1:Ny,kk=1:Nz]
##
k_guess
##

using Roots
kz = Roots.find_zero(k -> _solve_Δω²(k,ω_mpb,ε⁻¹_SHM3,ds;neigs=1,eigind=1,maxiter=3000,tol=1e-7), ds.k[3], Roots.Newton())

_solve_Δω²(k,ω_mpb,ε⁻¹_SHM3,ds;neigs=1,eigind=1,maxiter=3000,tol=1e-7)

##
H,k = solve_k(ω_mpb,eps)

# @btime H,k = solve_k(0.66,$eps) # 998.256 ms (43540 allocations: 240.61 MiB)
# @btime H,k = solve_k(0.62,ε⁻¹_mpb,ds)    #  1.150 ms (222 allocations: 5.17 MiB)
@btime H,k = solve_k(0.62,ε⁻¹_mpb,ds)
@profiler H,k = solve_k(0.66,eps)
@profiler H,k = solve_k(0.64,ε⁻¹_mpb,ds)

# Zygote.@profile solve_k(0.66,eps)
# Zygote.@profile solve_k(0.64,ε⁻¹_mpb,ds)

##
@btime H,k = solve_k(0.66,$eps)
#

##

# calculate ng

# function foo(H::Array{Complex{Float64},2},k::Float64,ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ω::Float64)::Float64
# 	kpg = kpG(k,ds.grid::MaxwellGrid)::Array{KVec,3}
#     ng = ω /  real( ( H[:,1]' * Mₖ(H[:,1], ε⁻¹,kpg) )[1])
# end

# @profiler foo(H,k,ε⁻¹_mpb,ω_mpb)
# @btime foo($H,$k,$ε⁻¹_mpb,$ω_mpb)
# # calculate
# @benchmark Zygote.gradient(foo,$H,$k,$ε⁻¹_mpb,$ω_mpb)

# Zygote.gradient(foo,H,k,ε⁻¹_mpb,ω_mpb)

# @profiler Zygote.gradient(foo,H,k,ε⁻¹_mpb,ω_mpb)

# @profiler H̄,k̄,ε̄⁻¹,ω̄ = Zygote.gradient(foo,H,k,ε⁻¹_mpb,ω_mpb)

# Zygote.@profile foo(H,k,ε⁻¹_mpb,ω_mpb)

##

H,k = solve_k(0.62,ε⁻¹_mpb;ds)
# @btime H,k = solve_k(0.62,ε⁻¹_mpb,ds)    #  1.150 ms (222 allocations: 5.17 MiB)
nᵧ, nᵧ_back = Zygote.pullback(foo,H,ω_mpb,ε⁻¹_mpb)
H̄,ω̄,ε̄⁻¹ = nᵧ_back(1)

## calculate ng from initial modesolve inputs

# function goo(H,ω,ε⁻¹)
# 	H,k = solve_k(ω,ε⁻¹;ds)
#     ng = ω /  real( ( H[:,1]' * Mₖ(H[:,1], ε⁻¹,ds) )[1])
# end

function Mop(Hin::AbstractArray{ComplexF64,4},kpG::AbstractArray{KVec,3})::AbstractArray{ComplexF64,4}
    d = zcross_t2c(Hin,kpG);
    # e = ifft(ε⁻¹_dot(d,ε⁻¹),(2:4));
    kcross_c2t(d,kpG)
end

# function goo(H::Array{Complex{Float64},2},k::Float64,ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ω::Float64)::Float64
function goo(H::Array{Complex{Float64},2},k::Float64)::Float64
	# H,k = solve_k(ω,ε⁻¹;ds)
	kpg = kpG(k,ds.grid::MaxwellGrid)::Array{KVec,3}
	# ng = ω /  real( ( H[:,1]' * Mop(H[:,1], kpg) )[1])
	out = ( H[:,1]' * Mop(H[:,1], kpg) )[1]
end

@profiler goo(H,k,ε⁻¹_mpb,ω_mpb)



## Define differentiable versions of the functions used to calculate ng


# Non-mutating, all AbstractArray Operators

function KplusG(kz,gx,gy,gz)
	# scale = ds.kpG[i,j,k].mag
	kpg = [-gx; -gy; kz-gz]
	mag = sqrt(sum(abs2.(kpg)))
	if mag==0
		n = [0.; 1.; 0.] # SVector(0.,1.,0.)
		m = [0.; 0.; 1.] # SVector(0.,0.,1.)
	else
		if kpg[1]==0. && kpg[2]==0.    # put n in the y direction if k+G is in z
			n = [0.; 1.; 0.] #SVector(0.,1.,0.)
		else                                # otherwise, let n = z x (k+G), normalized
			temp = [0.; 0.; 1.] × kpg  #SVector(0.,0.,1.) × kpg
			n = temp / sqrt(sum(abs2.(temp)))
		end
	end
	# m = n x (k+G), normalized
	mtemp = n × kpg
	m = mtemp / sqrt(sum(abs2.(mtemp))) #sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
	return kpg, mag, m, n
end

function new_t2c(Hin,kz,gx,gy,gz)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,3,Nx,Ny,Nz) # Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
		Hout[1,i,j,k] = ( Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * n[1] ) * mag
        Hout[2,i,j,k] = ( Hin[1,i,j,k] * m[2] + Hin[2,i,j,k] * n[2] ) * mag
        Hout[3,i,j,k] = ( Hin[1,i,j,k] * m[3] + Hin[2,i,j,k] * n[3] ) * mag
    end
    return Hout
end

function new_c2t(Hin,kz,gx,gy,gz)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,2,Nx,Ny,Nz) # Zygote.Buffer(Hin,2,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] =  Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * m[2] + Hin[3,i,j,k] * m[3]
        Hout[2,i,j,k] =  Hin[1,i,j,k] * n[1] + Hin[2,i,j,k] * n[2] + Hin[3,i,j,k] * n[3]
    end
    return Hout
end

function new_zcross_t2c(Hin,kz,gx,gy,gz)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,3,Nx,Ny,Nz) # Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] = -Hin[1,i,j,k] * m[2] - Hin[2,i,j,k] * n[2]
        Hout[2,i,j,k] =  Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * n[1]
        Hout[3,i,j,k] = 0.0
    end
    return Hout
end

function new_kcross_t2c(Hin,kz,gx,gy,gz)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,3,Nx,Ny,Nz) # Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] = ( Hin[1,i,j,k] * n[1] - Hin[2,i,j,k] * m[1] ) * -mag
        Hout[2,i,j,k] = ( Hin[1,i,j,k] * n[2] - Hin[2,i,j,k] * m[2] ) * -mag
        Hout[3,i,j,k] = ( Hin[1,i,j,k] * n[3] - Hin[2,i,j,k] * m[3] ) * -mag
    end
    return Hout
end

function new_kcross_c2t(Hin,kz,gx,gy,gz)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,2,Nx,Ny,Nz) # Zygote.Buffer(Hin,2,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
        at1 = Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * m[2] + Hin[3,i,j,k] * m[3]
        at2 = Hin[1,i,j,k] * n[1] + Hin[2,i,j,k] * n[2] + Hin[3,i,j,k] * n[3]
        Hout[1,i,j,k] =  -at2 * mag
        Hout[2,i,j,k] =  at1 * mag
    end
    return Hout
end

function new_kcrossinv_t2c(Hin,kz,gx,gy,gz)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] = ( Hin[1,i,j,k] * n[1] - Hin[2,i,j,k] * m[1] ) / mag
        Hout[2,i,j,k] = ( Hin[1,i,j,k] * n[2] - Hin[2,i,j,k] * m[2] ) / mag
        Hout[3,i,j,k] = ( Hin[1,i,j,k] * n[3] - Hin[2,i,j,k] * m[3] ) / mag
    end
    return Hout
end

function new_kcrossinv_c2t(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(2,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,2,Nx,Ny,Nz) # Zygote.Buffer(Hin,2,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
        scale = -1 / mag
        at1 = Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * m[2] + Hin[3,i,j,k] * m[3]
        at2 = Hin[1,i,j,k] * n[1] + Hin[2,i,j,k] * n[2] + Hin[3,i,j,k] * n[3]
        Hout[1,i,j,k] =  at2 / mag
        Hout[2,i,j,k] =  -at1 / mag
    end
    return Hout
end

function new_ε⁻¹_dot(Hin,ε⁻¹)
    # Hout = similar(Hin)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,3,Nx,Ny,Nz) # Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
        Hout[1,i,j,k] =  ε⁻¹[1,1,i,j,k]*Hin[1,i,j,k] + ε⁻¹[2,1,i,j,k]*Hin[2,i,j,k] + ε⁻¹[3,1,i,j,k]*Hin[3,i,j,k]
        Hout[2,i,j,k] =  ε⁻¹[1,2,i,j,k]*Hin[1,i,j,k] + ε⁻¹[2,2,i,j,k]*Hin[2,i,j,k] + ε⁻¹[3,2,i,j,k]*Hin[3,i,j,k]
        Hout[3,i,j,k] =  ε⁻¹[1,3,i,j,k]*Hin[1,i,j,k] + ε⁻¹[2,3,i,j,k]*Hin[2,i,j,k] + ε⁻¹[3,3,i,j,k]*Hin[3,i,j,k]
        # Hout[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][1,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][1,3]*Hin[3,i,j,k]
        # Hout[2,i,j,k] =  ε⁻¹[i,j,k][2,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[3,i,j,k]
        # Hout[3,i,j,k] =  ε⁻¹[i,j,k][3,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
    end
    return Hout
end

function new_ε_dot_approx(Hin,ε⁻¹)
    # Hout = similar(Hin)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,3,Nx,Ny,Nz) # Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
        ε_ave = 3 / tr(ε⁻¹[:,:,i,j,k])
        Hout[1,i,j,k] =  ε_ave * Hin[1,i,j,k]
        Hout[2,i,j,k] =  ε_ave * Hin[2,i,j,k]
        Hout[3,i,j,k] =  ε_ave * Hin[3,i,j,k]
    end
    return Hout
end

function new_M(Hin,ε⁻¹,kz,gx,gy,gz)
    d = fft(new_kcross_t2c(Hin,kz,gx,gy,gz),(2:4));
    e = ifft(new_ε⁻¹_dot(d,ε⁻¹),(2:4)); # (-1/(π)) .*
    new_kcross_c2t(e,kz,gx,gy,gz)
end

function new_Mₖ(Hin,ε⁻¹,kz,gx,gy,gz)
    d = fft(new_zcross_t2c(Hin,kz,gx,gy,gz),(2:4));
    e = ifft(new_ε⁻¹_dot(d,ε⁻¹),(2:4)); # (-1/(π)) .*
    new_kcross_c2t(e,kz,gx,gy,gz)
end


# Zygote versions

function sum2(op,arr)
    return sum(op,arr)
end

function sum2adj( Δ, op, arr )
    n = length(arr)
    g = x->Δ*Zygote.gradient(op,x)[1]
    return ( nothing, map(g,arr))
end

Zygote.@adjoint function sum2(op,arr)
    return sum2(op,arr),Δ->sum2adj(Δ,op,arr)
end
Zygote.refresh()

function zyg_KplusG(kz,gx,gy,gz)
	# scale = ds.kpG[i,j,k].mag
	kpg = [-gx; -gy; kz-gz]
	mag = sqrt(sum2(abs2,kpg))
	if mag==0
		n = [0.; 1.; 0.] # SVector(0.,1.,0.)
		m = [0.; 0.; 1.] # SVector(0.,0.,1.)
	else
		if kpg[1]==0. && kpg[2]==0.    # put n in the y direction if k+G is in z
			n = [0.; 1.; 0.] #SVector(0.,1.,0.)
		else                                # otherwise, let n = z x (k+G), normalized
			ntemp = [0.; 0.; 1.] × kpg  #SVector(0.,0.,1.) × kpg
			n = ntemp / sqrt(sum2(abs2,ntemp))
		end
	end
	# m = n x (k+G), normalized
	mtemp = n × kpg
	m = mtemp / sqrt(sum2(abs2,mtemp)) #sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
	return kpg, mag, m, n
end

function zyg_t2c(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(3,size(Hin)[2:end]...))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = zyg_KplusG(kz,gx[i],gy[j],gz[k])
		Hout[1,i,j,k] = ( Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * n[1] ) * mag
        Hout[2,i,j,k] = ( Hin[1,i,j,k] * m[2] + Hin[2,i,j,k] * n[2] ) * mag
        Hout[3,i,j,k] = ( Hin[1,i,j,k] * m[3] + Hin[2,i,j,k] * n[3] ) * mag
    end
    return copy(Hout)
end

function zyg_c2t(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(2,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,2,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = zyg_KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] =  Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * m[2] + Hin[3,i,j,k] * m[3]
        Hout[2,i,j,k] =  Hin[1,i,j,k] * n[1] + Hin[2,i,j,k] * n[2] + Hin[3,i,j,k] * n[3]
    end
    return copy(Hout)
end

function zyg_zcross_t2c(Hin,kz,gx,gy,gz)
    # Hout = zeros(ComplexF64,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = zyg_KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] = -Hin[1,i,j,k] * m[2] - Hin[2,i,j,k] * n[2]
        Hout[2,i,j,k] =  Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * n[1]
        Hout[3,i,j,k] = 0.0
    end
    return copy(Hout)
end

function zyg_kcross_t2c(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = zyg_KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] = ( Hin[1,i,j,k] * n[1] - Hin[2,i,j,k] * m[1] ) * -mag
        Hout[2,i,j,k] = ( Hin[1,i,j,k] * n[2] - Hin[2,i,j,k] * m[2] ) * -mag
        Hout[3,i,j,k] = ( Hin[1,i,j,k] * n[3] - Hin[2,i,j,k] * m[3] ) * -mag
    end
    return copy(Hout)
end

# Hout = [ (kpg, mag, m, n = zyg_KplusG(kz,gx[i],gy[j],gz[k]); Hi[]) ]

function zyg_kcross_c2t(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(2,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,2,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = zyg_KplusG(kz,gx[i],gy[j],gz[k])
        at1 = Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * m[2] + Hin[3,i,j,k] * m[3]
        at2 = Hin[1,i,j,k] * n[1] + Hin[2,i,j,k] * n[2] + Hin[3,i,j,k] * n[3]
        Hout[1,i,j,k] =  -at2 * mag
        Hout[2,i,j,k] =  at1 * mag
    end
    return copy(Hout)
end

function zyg_kcrossinv_t2c(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = zyg_KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] = ( Hin[1,i,j,k] * n[1] - Hin[2,i,j,k] * m[1] ) / mag
        Hout[2,i,j,k] = ( Hin[1,i,j,k] * n[2] - Hin[2,i,j,k] * m[2] ) / mag
        Hout[3,i,j,k] = ( Hin[1,i,j,k] * n[3] - Hin[2,i,j,k] * m[3] ) / mag
    end
    return copy(Hout)
end

function zyg_kcrossinv_c2t(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(2,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,2,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = zyg_KplusG(kz,gx[i],gy[j],gz[k])
        scale = -1 / mag
        at1 = Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * m[2] + Hin[3,i,j,k] * m[3]
        at2 = Hin[1,i,j,k] * n[1] + Hin[2,i,j,k] * n[2] + Hin[3,i,j,k] * n[3]
        Hout[1,i,j,k] =  at2 / mag
        Hout[2,i,j,k] =  -at1 / mag
    end
    return copy(Hout)
end

function zyg_ε⁻¹_dot(Hin,ε⁻¹)
    # Hout = similar(Hin)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
        Hout[1,i,j,k] =  ε⁻¹[1,1,i,j,k]*Hin[1,i,j,k] + ε⁻¹[2,1,i,j,k]*Hin[2,i,j,k] + ε⁻¹[3,1,i,j,k]*Hin[3,i,j,k]
        Hout[2,i,j,k] =  ε⁻¹[1,2,i,j,k]*Hin[1,i,j,k] + ε⁻¹[2,2,i,j,k]*Hin[2,i,j,k] + ε⁻¹[3,2,i,j,k]*Hin[3,i,j,k]
        Hout[3,i,j,k] =  ε⁻¹[1,3,i,j,k]*Hin[1,i,j,k] + ε⁻¹[2,3,i,j,k]*Hin[2,i,j,k] + ε⁻¹[3,3,i,j,k]*Hin[3,i,j,k]
        # Hout[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][1,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][1,3]*Hin[3,i,j,k]
        # Hout[2,i,j,k] =  ε⁻¹[i,j,k][2,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[3,i,j,k]
        # Hout[3,i,j,k] =  ε⁻¹[i,j,k][3,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
    end
    return copy(Hout)
end

function zyg_ε_dot_approx(Hin,ε⁻¹)
    # Hout = similar(Hin)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    @inbounds for i=1:Nx,j=1:Ny,k=1:Nz
        ε_ave = 3 / tr(ε⁻¹[:,:,i,j,k])
        Hout[1,i,j,k] =  ε_ave * Hin[1,i,j,k]
        Hout[2,i,j,k] =  ε_ave * Hin[2,i,j,k]
        Hout[3,i,j,k] =  ε_ave * Hin[3,i,j,k]
    end
    return copy(Hout)
end

function zyg_M(Hin,ε⁻¹,kz,gx,gy,gz)
    d = fft(zyg_kcross_t2c(Hin,kz,gx,gy,gz),(2:4));
    e = ifft(zyg_ε⁻¹_dot(d,ε⁻¹),(2:4)); # (-1/(π)) .*
    zyg_kcross_c2t(e,kz,gx,gy,gz)
end

# function zyg_M(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
#     HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
#     HoutA = M(HinA,ε⁻¹,ds)
#     return vec(HoutA)
# end

function zyg_Mₖ(Hin,ε⁻¹,kz,gx,gy,gz)
    d = fft(zyg_zcross_t2c(Hin,kz,gx,gy,gz),(2:4));
    e = ifft(zyg_ε⁻¹_dot(d,ε⁻¹),(2:4));
    zyg_kcross_c2t(e,kz,gx,gy,gz)
end

# function new_Mₖ(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},kpG::AbstractArray{KVec,3})::Array{ComplexF64,1}
#     Nx,Ny,Nz = size(ε⁻¹)
#     HinA = reshape(Hin,(2,Nx,Ny,Nz))
#     HoutA = Mₖ(HinA,ε⁻¹,kpG)
#     return -vec(HoutA)
# end


##

function ng_fm(kz::Real)::Real
    # return real(dot(vec(Hin), vec(new_Mₖ(Hin,eps,kz,gx,gy,gz))))
    return real(dot(vec(Hin), vec(new_Mₖ(Hin,eps,1.55,gx,gy,gz)))) * kz
end

function zyg_ng(H,eps,kz)
	return real(dot(vec(H), vec(zyg_Mₖ(H,eps,kz,gx,gy,gz))))
end

function zyg_ng2(kz)
	return real(dot(vec(Hin), vec(zyg_Mₖ(Hin,eps,kz,gx,gy,gz))))
end

##
#using ForwardDiff
ForwardDiff.derivative(ng_fm, kz)

# @benchmark fn(1.55)
##
# f2_inputs = Hin, epsC, kzCA
# f2_tape = ReverseDiff.GradientTape(f2, f2_inputs)
# f2_cfg = ReverseDiff.GradientConfig(f2, epsC)

#ReverseDiff.gradient(f2, (Hin, epsC, kzCA))
#@benchmark ReverseDiff.gradient(e -> f2(Hin,e,kzA),eps)
Zygote.gradient(zyg_ng,Hin,eps,1.55)

##

# function solve_k_pullback(H̄, k̄, ω, ε⁻¹) #;neigs,eigind,maxiter,tol) # ω̄ ₖ)
# 	H, kz = solve_k(ω, ε⁻¹) #;neigs,eigind,maxiter,tol) #Ω
#
# 	# hacky handling of non-differentiated parameters for now
# 	eigind = 1
# 	Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
# 	gx = fftfreq(Nx,Nx/6.0)
# 	gy = fftfreq(Ny,Ny/4.0)
# 	gz = fftfreq(Nz,Nz/1.0)
# 	## end hacky parameter handling
#
# 	P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
# 	A = M̂(ε⁻¹,kz,gx,gy,gz) - ω^2 * I
# 	b = ( I  -  P ) * H̄[:,eigind]
# 	λ⃗₀ = IterativeSolvers.bicgstabl(A,b,3)
# 	ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
# 	# Hₖ =  ( I  -  P ) * * ( Mₖ(H[:,eigind], ε⁻¹,ds) / ω )
# 	ω̄  =  ωₖ * k̄
# 	λ⃗₀ -= P*λ⃗₀ - ω̄  * H
# 	Ha = reshape(H,(2,ds.Nx,ds.Ny,ds.Nz))
# 	Ha_F =  ds.𝓕 * kcross_t2c(Ha,ds.kpG)
# 	λ₀ = reshape(λ⃗₀,(2,ds.Nx,ds.Ny,ds.Nz))
# 	λ₀_F  = ds.𝓕 * kcross_t2c(λ₀,ds.kpG)
# 	# ε̄ ⁻¹ = ( ds.𝓕 * kcross_t2c(λ₀,ds) ) .* ( ds.𝓕 * kcross_t2c(Ha,ds) )
# 	ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) ) for i=1:ds.Nx,j=1:ds.Ny,kk=1:ds.Nz]
# 	return ω̄ , ε⁻¹_bar
# end

# Zygote.@adjoint solve_k(ω,ε⁻¹) = solve_k(ω,ε⁻¹), (H̄, k̄ ) -> solve_k_pullback(H̄, k̄, ω,ε⁻¹)

Zygote.@adjoint function solve_k(ω,ε⁻¹)
	H, kz = solve_k(ω,ε⁻¹)
	(H, kz), (H̄, k̄ ) -> begin
		# hacky handling of non-differentiated parameters for now
		eigind = 1
		Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
		gx = fftfreq(Nx,Nx/6.0)
		gy = fftfreq(Ny,Ny/4.0)
		gz = fftfreq(Nz,Nz/1.0)
		## end hacky parameter handling

		P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
		A = M̂(ε⁻¹,kz,gx,gy,gz) - ω^2 * I
		b = ( I  -  P ) * H̄[:,eigind]
		λ⃗₀ = IterativeSolvers.bicgstabl(A,b,3)
		ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
		# Hₖ =  ( I  -  P ) * * ( Mₖ(H[:,eigind], ε⁻¹,ds) / ω )
		ω̄  =  ωₖ * k̄
		λ⃗₀ -= P*λ⃗₀ - ω̄  * H
		Ha = reshape(H,(2,Nx,Ny,Nz))
		Ha_F =  fft(kcross_t2c(Ha,kz,gx,gy,gz),(2:4))
		λ₀ = reshape(λ⃗₀,(2,Nx,Ny,Nz))
		λ₀_F  = fft(kcross_t2c(λ₀,kz,gx,gy,gz),(2:4))
		# ε̄ ⁻¹ = ( 𝓕 * kcross_t2c(λ₀,ds) ) .* ( 𝓕 * kcross_t2c(Ha,ds) )
		ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) ) for i=1:Nx,j=1:Ny,kk=1:Nz]
		return ω̄ , ε⁻¹_bar
	end
end

# Zygote.@adjoint function vcat(xs::Union{Number, AbstractVector}...)
#     vcat(xs...), dy -> begin
#         d = 0
#         map(xs) do x
#             x isa Number && return dy[d+=1]
#             l = length(x)
#             dx = dy[d+1:d+l]
#             d += l
#             return dx
#         end
#     end
# end

Zygote.refresh()

##
LinearAlgebra.ldiv!(c,A::LinearMaps.LinearMap,b) = mul!(c,A',b)

H, kz = solve_k(ω_mpb, ε⁻¹_mpb;ds)
H = H[:,1]
P = LinearMap(x -> H * dot(H,x),length(H),ishermitian=true)
# ⟂ = I  -  P #LinearMap(x -> H * dot(H,x),length(H),ishermitian=true)
A = M̂(ε⁻¹_mpb;ds) - ω_mpb^2 * I
println(typeof(A))
b = ( I  -  P ) * H̄
λ⃗₀ = IterativeSolvers.bicgstabl(A,b,2)
# Hₖ =  ( I  -  P ) * * ( Mₖ(H[:,eigind], ε⁻¹,ds) / ω )
eigind = 1
ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹_mpb,ds) )[1]) / ω_mpb # ω²ₖ / ( 2 * ω )
ω̄  =  ωₖ * k̄

λ⃗₀ -= P*λ⃗₀ - ω̄  * H
Ha = reshape(H,(2,ds.Nx,ds.Ny,ds.Nz))
λ₀ = reshape(λ⃗₀,(2,ds.Nx,ds.Ny,ds.Nz))
# ε̄ ⁻¹ = ( ds.𝓕 * kcross_t2c(λ₀,ds) ) .* ( ds.𝓕 * kcross_t2c(Ha,ds) )
ε⁻¹_bar = [ Diagonal( real.( ( ds.𝓕 * kcross_t2c(λ₀,ds.kpG) )[:,i,j,kk] .* ( ds.𝓕 * kcross_t2c(Ha,ds.kpG) )[:,i,j,kk] ) ) for i=1:ds.Nx,j=1:ds.Ny,kk=1:ds.Nz]





##
H̄ = rand(ComplexF64,size(ds.H⃗))[:,1]
k̄ = rand(Float64)
solve_k_pullback(H̄, k̄, ω_mpb, ε⁻¹_mpb;ds,neigs=1,eigind=1,maxiter=3000,tol=1e-8)

##

@code_typed solve_k(ω_mpb,ε⁻¹_mpb;ds)
##

Zygote.gradient(solve_k,ω_mpb,eps)
# Zygote.gradient(solve_k,ω_mpb,ε⁻¹_mpb,ds)
Zygote.pullback(solve_k,ω_mpb,eps)

##
function solve_k2(ω::Float64,ε⁻¹::Array{SHM3,3};neigs=1,eigind=1,maxiter=3000,tol=1e-8)
    kz = Roots.find_zero(k -> _solve_Δω²(k,ω,ε⁻¹,ds;neigs,eigind,maxiter,tol), ds.k[3], Roots.Newton())
    # ds.ω = √ds.ω²
    # ds.ωₖ = ds.ω²ₖ / ( 2 * ds.ω )
    return ds.H⃗, kz #, 1/ds.ωₖ
end


Juno.@enter Zygote.gradient((om,ep)->sum(abs2.(solve_k(om,ep)[1])),ω_mpb,eps)


Zygote.gradient(ω_mpb,ε⁻¹) do ω, εinv
	H,k = solve_k(ω, εinv)
	real( sum(abs2.(H)) + k )
end

##
# zyg_ng2(1.55)
# Zygote.gradient(zyg_ng2,1.55)
Zygote.refresh()
@benchmark Zygote.gradient(zyg_ng2,1.55)
##
# Zygote.gradient(H->zyg_ng(H,1.55),Hin)
@benchmark Zygote.gradient(e->zyg_ng($Hin,1.55,e),$eps)

##

Zygote.refresh()

@benchmark Zygote.gradient(zyg_ng,$Hin,1.55)

# @benchmark Zygote.gradient(f2,$Hin,$eps,1.55)

# BenchmarkTools.Trial:
#   memory estimate:  151.14 GiB
#   allocs estimate:  8524960
#   --------------
#   minimum time:     31.358 s (15.62% GC)
#   median time:      31.358 s (15.62% GC)
#   mean time:        31.358 s (15.62% GC)
#   maximum time:     31.358 s (15.62% GC)
#   --------------
#   samples:          1
#   evals/sample:     1


gx = collect(fftfreq(Nx,Nx/6.0))
gy = collect(fftfreq(Ny,Ny/4.0))
gz = collect(fftfreq(Nz,Nz/1.0))
## end hacky parameter handling

eigind=1
kz = k
P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
A = M̂(ε⁻¹,kz,gx,gy,gz) - ω^2 * I
b = ( I  -  P ) * H̄[:,eigind]
λ⃗₀ = IterativeSolvers.bicgstabl(A,b,3)
ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
# Hₖ =  ( I  -  P ) * * ( Mₖ(H[:,eigind], ε⁻¹,ds) / ω )
ω̄  =  ωₖ * real(k̄)
λ⃗₀ -= P*λ⃗₀ - ω̄  * H
Ha = reshape(H,(2,Nx,Ny,Nz))
Ha_F =  fft(kcross_t2c(Ha,kz,gx,gy,gz),(2:4))
λ₀ = reshape(λ⃗₀,(2,Nx,Ny,Nz))
λ₀_F  = fft(kcross_t2c(λ₀,kz,gx,gy,gz),(2:4))
# ε̄ ⁻¹ = ( 𝓕 * kcross_t2c(λ₀,ds) ) .* ( 𝓕 * kcross_t2c(Ha,ds) )
ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) ) for i=1:Nx,j=1:Ny,kk=1:Nz]

ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) )[a,b] for a=1:3,b=1:3,i=1:Nx,j=1:Ny,kk=1:Nz]
