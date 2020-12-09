##
# import Base: +, -, zero
import Base.Iterators
using Coverage, LinearAlgebra, StaticArrays, Zygote

# struct Point
#   x::Float64
#   y::Float64
# end
#
# width(p::Point) = p.x
# height(p::Point) = p.y
#
# a::Point + b::Point = Point(width(a) + width(b), height(a) + height(b))
# a::Point - b::Point = Point(width(a) - width(b), height(a) - height(b))
# dist(p::Point) = sqrt(width(p)^2 + height(p)^2)
#
# ##

@Zygote.adjoint (T :: Type{<:SVector})(xs :: Number ...) = T(xs...), dv -> (nothing, dv...)
@Zygote.adjoint (T :: Type{<:SVector})(x :: AbstractVector) = T(x), dv -> (nothing, dv)

@Zygote.adjoint enumerate(xs) = enumerate(xs), diys -> (map(last, diys),)

_ndims(::Base.HasShape{d}) where {d} = d
_ndims(x) = Base.IteratorSize(x) isa Base.HasShape ? _ndims(Base.IteratorSize(x)) : 1

@Zygote.adjoint function Iterators.product(xs...)
                    d = 1
                    Iterators.product(xs...), dy -> ntuple(length(xs)) do n
                        nd = _ndims(xs[n])
                        dims = ntuple(i -> i<d ? i : i+nd, ndims(dy)-nd)
                        d += nd
                        func = sum(y->y[n], dy; dims=dims)
                        ax = axes(xs[n])
                        reshape(func, ax)
                    end
                end

# @Zygote.adjoint width(p::Point) = p.x, x̄ -> (Point(x̄, 0),)
# @Zygote.adjoint height(p::Point) = p.y, ȳ -> (Point(0, ȳ),)
# @Zygote.adjoint Point(a, b) = Point(a, b), p̄ -> (p̄.x, p̄.y)
# zero(::Point) = Point(0, 0)

# ##
# xs = Point.(1:5, 5:9)
#
# function something(xs)
#     sum([ width(p1) + height(p2) for p1 in xs, p2 in xs ])
# end
#
# gradient(something, xs)

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

# kpg, mag, m, n = KpG(1.5,3.5,6.5,0.0)
# KpG2(1.5,3.5,6.5,0.0)
# KpG(1.5,3.5,6.5,0.0)[2]
# KpG2(1.5,3.5,6.5,0.0)[2]
# Zygote.gradient((kz,gx,gy,gz)->KpG(kz,gx,gy,gz)[2],1.5,3.5,6.5,0.0)
# Zygote.gradient((kz,gx,gy,gz)->KpG2(kz,gx,gy,gz)[2],1.5,3.5,6.5,0.0)

function ∇KpG2(a,b,c,d)
	Zygote.gradient((kz,gx,gy,gz)->KpG(kz,gx,gy,gz)[2],a,b,c,d)
end

function ∇KpG22(a::Float64,b::Float64,c::Float64,d::Float64)
	Zygote.gradient((kz::Float64,gx::Float64,gy::Float64,gz::Float64)->KpG2(kz,gx,gy,gz)[2],a,b,c,d)
end

using ReverseDiff
function ∇KpG23(a::Float64,b::Float64,c::Float64,d::Float64)
	# ReverseDiff.gradient(input->KpG(input...)[2],[a,b,c,d])
	ReverseDiff.gradient!(results, compiled_f_tape, [a,b,c,d])
end


using ForwardDiff

ForwardDiff.gradient(input->KpG(input...)[2],[1.5,3.5,6.5,0.0])

function ∇KpG24(a::Real,b::Real,c::Real,d::Real)
	ForwardDiff.gradient(input->KpG(input...)[2],[a,b,c,d])
end

# function ∇KpG25(a::Real,b::Real,c::Real,d::Real)
# 	Zygote.gradient(a,b,c,d) do kz,gx,gy,gz
# 		Zygote.forwarddiff([kz,gx,gy,gz]) do (kz,gx,gy,gz)
# 			KpG(kz,gx,gy,gz)[2]
# 		end
# 	end
# end

∇KpG25(a,b,c,d) = Zygote.gradient(a,b,c,d) do kz,gx,gy,gz
	Zygote.forwarddiff([kz,gx,gy,gz]) do (kz,gx,gy,gz)
		KpG(kz,gx,gy,gz)[2]
	end
end


∇KpG2(1.5,3.5,6.5,0.0)
∇KpG22(1.5,3.5,6.5,0.0)
∇KpG23(1.5,3.5,6.5,0.0)
∇KpG24(1.5,3.5,6.5,0.0)
∇KpG25(1.5,3.5,6.5,0.0)


##
@btime ∇KpG2(1.5,3.5,6.5,0.0)
@btime ∇KpG22(1.5,3.5,6.5,0.0)
@btime ∇KpG23(1.5,3.5,6.5,0.0)
@btime ∇KpG24(1.5,3.5,6.5,0.0)
@btime ∇KpG25(1.5,3.5,6.5,0.0)
##

ReverseDiff.gradient(input->KpG(input...)[2],[1.5,3.5,6.5,0.0])

const f_tape = ReverseDiff.GradientTape(input->KpG(input...)[2], rand(4))
const compiled_f_tape = ReverseDiff.compile(f_tape)
results = similar(rand(4))
@btime ReverseDiff.gradient!(results, compiled_f_tape, [1.5,3.5,6.5,0.0])

##
using Tullio
Nz = 1
gx = collect(fftfreq(Nx,Nx/6.0))
gy = collect(fftfreq(Ny,Ny/4.0))
gz = collect(fftfreq(Nz,Nz/1.0))
kz = 1.476254662632502 #1.5
kpg_tuples = [KpG(kz,ggx,ggy,ggz) for ggx in gx, ggy in gy, ggz in gz]
kpg = [kpg_tuples[i,j,k][1][a] for a=1:3,i=1:Nx,j=1:Ny,k=1:Nz]
kpg_mag = [kpg_tuples[i,j,k][2] for i=1:Nx,j=1:Ny,k=1:Nz]
m = [kpg_tuples[i,j,k][3][a] for a=1:3,i=1:Nx,j=1:Ny,k=1:Nz]
n = [kpg_tuples[i,j,k][4][a] for a=1:3,i=1:Nx,j=1:Ny,k=1:Nz]
mn = [kpg_tuples[i,j,k][2+b][a] for a=1:3,b=1:2,i=1:Nx,j=1:Ny,k=1:Nz]
Ha = copy(reshape(H,(2,Nx,Ny,1)))
Hri = vcat(reim(reshape(Ha,(1,size(Ha)...)))...)
ε⁻¹


zxinds = [2; 1; 3]
zxscales = [-1.; 1.; 0.]
@tullio zxH[a,i,j,k] := zxscales[a] * Ha[b,i,j,k] * mn[zxinds[a],b,i,j,k] nograd=(zxscales,zxinds,m,n) verbose=true

using Zygote, ForwardDiff
function zx(Ha,mn)
	zxinds = [2; 1; 3]
	zxscales = [-1.; 1.; 0.]
	@tullio zxH[a,i,j,k] := zxscales[a] * Ha[b,i,j,k] * mn[zxinds[a],b,i,j,k] nograd=(zxscales,zxinds) #,m,n) #verbose=true # grad=Dual
end

zx(Ha,mn)
Zygote.gradient(abs2∘sum∘zx,Ha,mn)
@btime Zygote.gradient(abs2∘sum∘zx,$Ha,$mn)[1]

@btime fft(zx($Ha,$mn),(2:4))

@tullio e[a,i,j,k] :=  ε⁻¹[a,b,i,j,k] * fft(zx(Ha,mn),(2:4))[b,i,j,k] verbose=true

function ε⁻¹_op(ε⁻¹,Ha,mn)
	@tullio e[a,i,j,k] :=  ε⁻¹[a,b,i,j,k] * fft(zx(Ha,mn),(2:4))[b,i,j,k]
	return ifft(e,(2:4))
end

E = ε⁻¹_op(ε⁻¹,Ha,mn)
Zygote.gradient(abs2∘sum∘ε⁻¹_op,ε⁻¹,Ha,mn)
@btime Zygote.gradient(abs2∘sum∘ε⁻¹_op,$ε⁻¹,$Ha,$mn)

kxinds = [2; 1]
kxscales = [-1.; 1.]
@tullio kxE[b,i,j,k] := kxscales[b] * kpg_mag[i,j,k] * E[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) verbose=true
@tullio kxE[b,i,j,k] := kxscales[b] * kpg_mag[i,j,k] * ε⁻¹_op(ε⁻¹,Ha,mn)[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) verbose=true

function Mk(Ha,ε⁻¹,mn,kpg_mag)
	kxinds = [2; 1]
	kxscales = [-1.; 1.]
	@tullio kxE[b,i,j,k] := kxscales[b] * kpg_mag[i,j,k] * ε⁻¹_op(ε⁻¹,Ha,mn)[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds)
end

@btime Mk($Ha,$ε⁻¹,$mn,$kpg_mag)
@tullio out[_] := conj.(Ha)[b,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * ε⁻¹_op(ε⁻¹,Ha,mn)[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds) verbose=true

function H_Mk_H(Ha,ε⁻¹,mn,kpg_mag)
	kxinds = [2; 1]
	kxscales = [-1.; 1.]
	@tullio out[_] := conj.(Ha)[b,i,j,k] * kxscales[b] * kpg_mag[i,j,k] * ε⁻¹_op(ε⁻¹,Ha,mn)[a,i,j,k] * mn[a,kxinds[b],i,j,k] nograd=(kxscales,kxinds)
	return abs(out[1])
end

H_Mk_H(Ha,ε⁻¹,mn,kpg_mag)
ω_mpb / H_Mk_H(Ha,ε⁻¹,mn,kpg_mag)
( ng_mpb - ( ω_mpb / H_Mk_H(Ha,ε⁻¹,mn,kpg_mag) ) ) / ng_mpb
Zygote.gradient(H_Mk_H,Ha,ε⁻¹,mn,kpg_mag)
@btime Zygote.gradient(H_Mk_H,$Ha,$ε⁻¹,$mn,$kpg_mag)
# 3.108 ms (37422 allocations: 7.25 MiB) !!!

# function kpg_mn(kpg,mag)
# 	if mag==0
# 		n = [0.; 1.; 0.]
# 		m = [0.; 0.; 1.]
# 	else
# 		if kpg[1]==0. && kpg[2]==0.    # put n in the y direction if k+G is in z
# 			n = [0.; 1.; 0.] #SVector(0.,1.,0.)
# 		else                                # otherwise, let n = z x (k+G), normalized
# 			ntemp = [0.; 0.; 1.] × kpg
# 			n = ntemp / sqrt(sum2(abs2,ntemp)) # norm(ntemp) #
# 		end
# 	end
# 	# m = n x (k+G), normalized
# 	mtemp = n × kpg
# 	m = mtemp / sqrt(sum2(abs2,mtemp))
# 	return hcat(m,n)
# end

Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
Δz = 1.0
gx = Zygote.@ignore [[ggx;0.0;0.0] for ggx in collect(fftfreq(Nx,Nx/Δx))]
gy = Zygote.@ignore [[0.0;ggy;0.0] for ggy in collect(fftfreq(Ny,Ny/Δy))]
gz = Zygote.@ignore [[0.0;0.0;ggz] for ggz in collect(fftfreq(Nz,Nz/Δz))]
g⃗ = Zygote.@ignore [ [gx;gy;gz] for gx in fftfreq(Nx,Nx/Δx), gy in fftfreq(Ny,Ny/Δy), gz in fftfreq(Nz,Nz/Δz)]
g⃗ₜ_zero_mask = [ sum(abs2.(gg[1:2])) for gg in g⃗ ] .> 0.
g⃗ₜ_zero_mask! = .!(g⃗ₜ_zero_mask)
k⃗ = [0.;0.;kz]
@tullio kpg[a,i,j,k] := k⃗[a] - g⃗[i,j,k][a] nograd=g⃗ verbose=true
@tullio kpg_mag[i,j,k] := sqrt <| kpg[a,i,j,k]^2 verbose=true
zxinds = [2; 1; 3]
zxscales = [-1; 1. ;0.] #[[0. -1. 0.]; [-1. 0. 0.]; [0. 0. 0.]]
ŷ = [0.; 1. ;0.]
# @tullio kpg_nt[a,i,j,k] := zxscales[a] * kpg[zxinds[a],i,j,k] nograd=(zxscales,zxinds) verbose=true
@tullio kpg_nt[a,i,j,k] := zxscales[a] * kpg[zxinds[a],i,j,k] * g⃗ₜ_zero_mask[i,j,k] + ŷ[a] * g⃗ₜ_zero_mask![i,j,k]  nograd=(zxscales,zxinds,ŷ,g⃗ₜ_zero_mask,g⃗ₜ_zero_mask!) verbose=true
@tullio kpg_nmag[i,j,k] := sqrt <| kpg_nt[a,i,j,k]^2  verbose=true
@tullio kpg_n[a,i,j,k] := kpg_nt[a,i,j,k] / kpg_nmag[i,j,k]
xinds1 = [2; 3; 1]
xinds2 = [3; 1; 2]
@tullio kpg_mt[a,i,j,k] := kpg_n[xinds1[a],i,j,k] * kpg[xinds2[a],i,j,k] - kpg[xinds1[a],i,j,k] * kpg_n[xinds2[a],i,j,k] nograd=(xinds1,xinds2) verbose=true
@tullio kpg_mmag[i,j,k] := sqrt <| kpg_mt[a,i,j,k]^2  verbose=true
@tullio kpg_m[a,i,j,k] := kpg_mt[a,i,j,k] / kpg_mmag[i,j,k]
kpg_mn_basis = [[1. 0.] ; [0. 1.]]
@tullio kpg_mn[a,b,i,j,k] := kpg_mn_basis[b,1] * kpg_m[a,i,j,k] + kpg_mn_basis[b,2] * kpg_n[a,i,j,k] nograd=kpg_mn_basis verbose=true

function calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
	g⃗ = Zygote.@ignore [ [gx;gy;gz] for gx in fftfreq(Nx,Nx/Δx), gy in fftfreq(Ny,Ny/Δy), gz in fftfreq(Nz,Nz/Δz)]
	g⃗ₜ_zero_mask = Zygote.@ignore [ sum(abs2.(gg[1:2])) for gg in g⃗ ] .> 0.
	g⃗ₜ_zero_mask! = Zygote.@ignore .!(g⃗ₜ_zero_mask)
	ŷ = [0.; 1. ;0.]
	k⃗ = [0.;0.;kz]
	@tullio kpg[a,i,j,k] := k⃗[a] - g⃗[i,j,k][a] nograd=g⃗
	@tullio kpg_mag[i,j,k] := sqrt <| kpg[a,i,j,k]^2
	zxinds = [2; 1; 3]
	zxscales = [-1; 1. ;0.] #[[0. -1. 0.]; [-1. 0. 0.]; [0. 0. 0.]]
	@tullio kpg_nt[a,i,j,k] := zxscales[a] * kpg[zxinds[a],i,j,k] * g⃗ₜ_zero_mask[i,j,k] + ŷ[a] * g⃗ₜ_zero_mask![i,j,k]  nograd=(zxscales,zxinds,ŷ,g⃗ₜ_zero_mask,g⃗ₜ_zero_mask!)
	@tullio kpg_nmag[i,j,k] := sqrt <| kpg_nt[a,i,j,k]^2
	@tullio kpg_n[a,i,j,k] := kpg_nt[a,i,j,k] / kpg_nmag[i,j,k]
	xinds1 = [2; 3; 1]
	xinds2 = [3; 1; 2]
	@tullio kpg_mt[a,i,j,k] := kpg_n[xinds1[a],i,j,k] * kpg[xinds2[a],i,j,k] - kpg[xinds1[a],i,j,k] * kpg_n[xinds2[a],i,j,k] nograd=(xinds1,xinds2)
	@tullio kpg_mmag[i,j,k] := sqrt <| kpg_mt[a,i,j,k]^2
	@tullio kpg_m[a,i,j,k] := kpg_mt[a,i,j,k] / kpg_mmag[i,j,k]
	kpg_mn_basis = [[1. 0.] ; [0. 1.]]
	@tullio kpg_mn[a,b,i,j,k] := kpg_mn_basis[b,1] * kpg_m[a,i,j,k] + kpg_mn_basis[b,2] * kpg_n[a,i,j,k] nograd=kpg_mn_basis
	return kpg_mag, kpg_mn
end

function ng1(ω,ε⁻¹,Δx,Δy,Δz)
	Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
	H,kz = solve_k(ω, ε⁻¹)
	kpg_mag, kpg_mn = calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
	Ha = copy(reshape(H,(2,Nx,Ny,Nz)))
	n_g = ω / H_Mk_H(Ha,ε⁻¹,kpg_mn,kpg_mag)
end

ng1(0.65,ε⁻¹,6.0,4.0,1.0)
@btime ng1(0.65,$ε⁻¹,6.0,4.0,1.0)
# 910.910 ms (152467 allocations: 237.08 MiB)
Zygote.gradient(ng1,0.65,ε⁻¹,6.0,4.0,1.0)
@btime Zygote.gradient(ng1,0.65,ε⁻¹,6.0,4.0,1.0)
# 4.695 s (49137386 allocations: 6.82 GiB)

function ng2(H,ω,ε⁻¹,kpg_mn,kpg_mag)
	Ha = copy(reshape(H,(2,Nx,Ny,Nz)))
	n_g = ω / H_Mk_H(Ha,ε⁻¹,kpg_mn,kpg_mag)
end

ng2(H,ω,ε⁻¹,kpg_mn,kpg_mag)
@btime ng2($H,0.65,$ε⁻¹,$kpg_mn,$kpg_mag)
# 1.526 ms (36971 allocations: 2.63 MiB)
Zygote.gradient(ng2,H,0.65,ε⁻¹,kpg_mn,kpg_mag)
@btime Zygote.gradient(ng2,$H,0.65,$ε⁻¹,$kpg_mn,$kpg_mag)
# 3.181 ms (37379 allocations: 7.44 MiB)

function ng3(Ha,ω,ε⁻¹,kpg_mn,kpg_mag)
	n_g = ω / H_Mk_H(Ha,ε⁻¹,kpg_mn,kpg_mag)
end

ng3(Ha,ω,ε⁻¹,kpg_mn,kpg_mag)
@btime ng3($Ha,0.65,$ε⁻¹,$kpg_mn,$kpg_mag)
# 1.596 ms (36962 allocations: 2.45 MiB)
Zygote.gradient(ng3,Ha,0.65,ε⁻¹,kpg_mn,kpg_mag)
@btime Zygote.gradient(ng3,$Ha,0.65,$ε⁻¹,$kpg_mn,$kpg_mag)
# 3.164 ms (37356 allocations: 7.25 MiB)

function ng4(ω,ε⁻¹,kpg_mn,kpg_mag)
	H,kz = solve_k(ω, ε⁻¹)
	Ha = copy(reshape(H,(2,Nx,Ny,Nz)))
	n_g = ω / H_Mk_H(Ha,ε⁻¹,kpg_mn,kpg_mag)
end

ng4(ω,ε⁻¹,kpg_mn,kpg_mag)
@btime ng4(0.65,$ε⁻¹,$kpg_mn,$kpg_mag)
# 971.820 ms (80838 allocations: 245.70 MiB)
Zygote.gradient(ng4,0.65,ε⁻¹,kpg_mn,kpg_mag)
@btime Zygote.gradient(ng4,0.65,$ε⁻¹,$kpg_mn,$kpg_mag)
# 5.081 s (48676432 allocations: 6.76 GiB)

function ng5(kz,H,ω,ε⁻¹,Δx,Δy,Δz)
	Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
	kpg_mag, kpg_mn = calc_kpg(kz,Δx,Δy,Δz,Nx,Ny,Nz)
	Ha = copy(reshape(H,(2,Nx,Ny,Nz)))
	n_g = ω / H_Mk_H(Ha,ε⁻¹,kpg_mn,kpg_mag)
end

ng5(kz,H,ω,ε⁻¹,6.0,4.0,1.0)
@btime ng5($kz,$H,0.65,$ε⁻¹,6.0,4.0,1.0)
# 2.655 ms (55445 allocations: 5.64 MiB)
Zygote.gradient(ng5,kz,H,0.65,ε⁻¹,6.0,4.0,1.0)
@btime Zygote.gradient(ng5,$kz,$H,0.65,$ε⁻¹,6.0,4.0,1.0)
# 6.204 ms (56614 allocations: 14.89 MiB)

@tullio ε_approx[i,j,k] := 3 / ε⁻¹[a,a,i,j,k]
@tullio d_approx[b,i,j,k] := zx(Ha,kpg_mn)[b,i,j,k] * 3 / ε⁻¹[a,a,i,j,k]

function kx_t2c(H,mn,kpg_mag)
	kxscales = [-1.; 1.]
    @tullio d[a,i,j,k] := kxscales[b] * H[b,i,j,k] * mn[a,b,i,j,k] * kpg_mag[i,j,k] nograd=kxscales
end



d⃗ = kx_t2c(Ha,kpg_mn,kpg_mag)
@btime @tullio e⃗[a,i,j,k] := d⃗[b,i,j,k] * ε⁻¹[a,b,i,j,k] # verbose=true
@btime @tullio e⃗[a,i,j,k] := d⃗[b,i,j,k] * ε⁻¹_mpb[i,j,k][a,b] # verbose=true

function ftest(d⃗)
	# @tullio e⃗[a,i,j,k] := d⃗[b,i,j,k] * ε⁻¹_mpb[i,j,k][a,b]
	@tullio e⃗[a,i,j,k] := d⃗[b,i,j,k] * ε⁻¹[a,b,i,j,k]
end

@btime ftest($d⃗)
# 344.180 μs (6154 allocations: 672.33 KiB)  # with StaticArrays ε⁻¹
# 94.073 μs (3 allocations: 288.13 KiB)  # with normal Array ε⁻¹


# function kx_t2c!(ds::MaxwellData,m::Array{Float64,4},n::Array{Float64,4},kpg_mag::Array{Float64,3})::AbstractArray{ComplexF64,4}
function kx_t2c!(ds::MaxwellData,mn::Array{Float64,5},kpg_mag::Array{Float64,3})::AbstractArray{ComplexF64,4}
	# kxscales = [-1.; 1.]
	# @tullio ds.d[a,i,j,k] = kxscales[b] * ds.H[b,i,j,k] * mn[a,b,i,j,k] * kpg_mag[i,j,k] nograd=kxscales
	# @tullio ds.d[a,i,j,k] = ( ds.H[2,i,j,k] * n[a,i,j,k] - ds.H[1,i,j,k] * m[a,i,j,k] ) * kpg_mag[i,j,k]
	@tullio ds.d[a,i,j,k] = ( ds.H[2,i,j,k] * mn[a,2,i,j,k] - ds.H[1,i,j,k] * mn[a,1,i,j,k] ) * kpg_mag[i,j,k]
	# @tullio ds.d[a,i,j,k] = ( ds.H[2,i,j,k] * n[a,i,j,k] - ds.H[1,i,j,k] * m[a,i,j,k] )
	# @tullio ds.d[a,i,j,k] *=  kpg_mag[i,j,k]

    # @inbounds for i=1:ds.Nx,j=1:ds.Ny,k=1:ds.Nz
    #     scale = -ds.kpG[i,j,k].mag #-ds.kpG[i,j,k].mag
    #     ds.d[1,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[1] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[1] ) * scale
    #     ds.d[2,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[2] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[2] ) * scale
    #     ds.d[3,i,j,k] = ( ds.H[1,i,j,k] * ds.kpG[i,j,k].n[3] - ds.H[2,i,j,k] * ds.kpG[i,j,k].m[3] ) * scale
    # end
    return ds.d
end

kx_t2c!(ds,mn,kpg_mag)
@btime kx_t2c!($ds,$mn,$kpg_mag)
kx_t2c!(ds,m,n,kpg_mag)
@btime kx_t2c!($ds,$m,$n,$kpg_mag)

##
eigind = 1
P = LinearMap(x -> H[:,eigind] * dot(H[:,eigind],x),length(H[:,eigind]),ishermitian=true)
A = M̂(ε⁻¹,mn,kpg_mag,𝓕) - ω^2 * I
b = ( I  -  P ) * H̄[:,eigind]
λ⃗₀ = IterativeSolvers.bicgstabl(A,b,l)
# ωₖ = real( ( H[:,eigind]' * Mₖ(H[:,eigind], ε⁻¹,kz,gx,gy,gz) )[1]) / ω # ds.ω²ₖ / ( 2 * ω )
Ha = reshape(H,(2,Nx,Ny,Nz))
ωₖ = H_Mₖ_H(Ha,ε⁻¹,mn,kpg_mag) / ω
# Hₖ =  ( I  -  P ) * * ( Mₖ(H[:,eigind], ε⁻¹,ds) / ω )
ω̄  =  ωₖ * k̄
λ⃗₀ -= P*λ⃗₀ - ω̄  * H
Ha_F = 𝓕 * kx_t2c(H,mn,kpg_mag) #fft(kcross_t2c(Ha,kz,gx,gy,gz),(2:4))
λ₀ = reshape(λ⃗₀,(2,Nx,Ny,Nz))
λ₀_F  = 𝓕 * kx_t2c(λ₀,mn,kpg_mag) #fft(kcross_t2c(λ₀,kz,gx,gy,gz),(2:4))
# ε̄ ⁻¹ = ( 𝓕 * kcross_t2c(λ₀,ds) ) .* ( 𝓕 * kcross_t2c(Ha,ds) )
# ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) ) for i=1:Nx,j=1:Ny,kk=1:Nz]
ε⁻¹_bar = [ Diagonal( real.(λ₀_F[:,i,j,kk] .* Ha_F[:,i,j,kk]) )[a,b] for a=1:3,b=1:3,i=1:Nx,j=1:Ny,kk=1:Nz]



##

function zyg_zcross_t2c(Hin,kz,gx,gy,gz)
    # Hout = zeros(ComplexF64,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = Zygote.forwarddiff(kz) do kz
                            zyg_KplusG(kz,gx[i],gy[j],gz[k])
                        end
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
		kpg, mag, m, n = Zygote.forwarddiff(kz) do kz
                            zyg_KplusG(kz,gx[i],gy[j],gz[k])
                        end
        Hout[1,i,j,k] = ( Hin[1,i,j,k] * n[1] - Hin[2,i,j,k] * m[1] ) * -mag
        Hout[2,i,j,k] = ( Hin[1,i,j,k] * n[2] - Hin[2,i,j,k] * m[2] ) * -mag
        Hout[3,i,j,k] = ( Hin[1,i,j,k] * n[3] - Hin[2,i,j,k] * m[3] ) * -mag
    end
    return copy(Hout)
end

function zyg_kcross_c2t(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(2,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,2,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = Zygote.forwarddiff(kz) do kz
                            zyg_KplusG(kz,gx[i],gy[j],gz[k])
                        end
        at1 = Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * m[2] + Hin[3,i,j,k] * m[3]
        at2 = Hin[1,i,j,k] * n[1] + Hin[2,i,j,k] * n[2] + Hin[3,i,j,k] * n[3]
        Hout[1,i,j,k] =  -at2 * mag
        Hout[2,i,j,k] =  at1 * mag
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
