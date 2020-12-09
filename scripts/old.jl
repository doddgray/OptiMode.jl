## old code from Maxwell.jl

########### PWE (k-space) grid initialization routines #################

function k_mn(k::SArray{Tuple{3},Float64,1,3})
    mag = sqrt(k[1]^2 + k[2]^2 + k[3]^2)
    if mag==0
        n = SVector(0.,1.,0.)
        m = SVector(0.,0.,1.)
    else
        if k[1]==0. && k[2]==0.    # put n in the y direction if k+G is in z
            n = SVector(0.,1.,0.)
        else                                # otherwise, let n = z x (k+G), normalized
            temp = SVector(0.,0.,1.) × k
            n = temp / sqrt( temp[1]^2 + temp[2]^2 + temp[3]^2 )
        end
    end

    # m = n x (k+G), normalized
    mtemp = n × k
    m = mtemp / sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
    return m,n
end

struct KVec
    k::SVector{3,Float64}   # 3-vector
    mag::Float64            # vector magnitude
    m::SVector{3,Float64}   # k vector normal with even parity in y
    n::SVector{3,Float64}   # k vector normal with even parity in x
end

KVec(k::SVector{3,Float64}) = KVec(
    k,
    sqrt(sum(abs2(ki) for ki in k)), # sqrt(sum(ki^2 for ki in k)),
    k_mn(k)...,
)

function kpG(k::SVector{3,Float64},g::MaxwellGrid)::Array{KVec,3}
    [KVec(k-gx-gy-gz) for gx=g.gx, gy=g.gy, gz=g.gz]
    # [KVec(k+gx+gy+gz) for gx=g.gx, gy=g.gy, gz=g.gz]
end

function kpG(kz::Float64,g::MaxwellGrid)::Array{KVec,3}
    # kpG(SVector(0.,0.,kz),g::MaxwellGrid)::Array{KVec,3}
    [KVec(SVector(0.,0.,kz)-gx-gy-gz) for gx=g.gx, gy=g.gy, gz=g.gz]
end

########### non-mutating operatators using StaticArrays #################

function KplusG(kz,gx,gy,gz)
	# scale = ds.kpG[i,j,k].mag
	kpg = [-gx; -gy; kz-gz]
	mag = norm(kpg) #sqrt(sum(abs2.(kpg)))
	if mag==0
		n = [0.; 1.; 0.] # SVector(0.,1.,0.)
		m = [0.; 0.; 1.] # SVector(0.,0.,1.)
	else
		if kpg[1]==0. && kpg[2]==0.    # put n in the y direction if k+G is in z
			n = [0.; 1.; 0.] #SVector(0.,1.,0.)
		else                                # otherwise, let n = z x (k+G), normalized
			ntemp = [0.; 0.; 1.] × kpg  #SVector(0.,0.,1.) × kpg
			n = ntemp / norm(ntemp) # sqrt(sum(abs2.(ntemp)))
		end
	end
	# m = n x (k+G), normalized
	mtemp = n × kpg
	m = mtemp / norm(mtemp) # sqrt(sum(abs2.(mtemp))) #sqrt( mtemp[1]^2 + mtemp[2]^2 + mtemp[3]^2 )
	return kpg, mag, m, n
end

function t2c(Hin,kz,gx,gy,gz)
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

function c2t(Hin,kz,gx,gy,gz)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = similar(Hin,2,Nx,Ny,Nz) # Zygote.Buffer(Hin,2,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = KplusG(kz,gx[i],gy[j],gz[k])
        Hout[1,i,j,k] =  Hin[1,i,j,k] * m[1] + Hin[2,i,j,k] * m[2] + Hin[3,i,j,k] * m[3]
        Hout[2,i,j,k] =  Hin[1,i,j,k] * n[1] + Hin[2,i,j,k] * n[2] + Hin[3,i,j,k] * n[3]
    end
    return Hout
end

function zcross_t2c(Hin,kz,gx,gy,gz)
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

function kcross_t2c(Hin,kz,gx,gy,gz)
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

function kcross_c2t(Hin,kz,gx,gy,gz)
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

function kcrossinv_t2c(Hin,kz,gx,gy,gz)
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

function kcrossinv_c2t(Hin,kz,gx,gy,gz)
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

function ε⁻¹_dot(Hin,ε⁻¹)
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

function ε_dot_approx(Hin,ε⁻¹)
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

function M(Hin::AbstractArray{T,4},ε⁻¹,kz,gx,gy,gz)::AbstractArray{T,4} where T
    d = fft(kcross_t2c(Hin,kz,gx,gy,gz),(2:4));
    e = ifft(ε⁻¹_dot(d,ε⁻¹),(2:4)); # (-1/(π)) .*
    kcross_c2t(e,kz,gx,gy,gz)
end

function Mₖ(Hin,ε⁻¹,kz,gx,gy,gz)
    d = fft(zcross_t2c(Hin,kz,gx,gy,gz),(2:4));
    e = ifft(ε⁻¹_dot(d,ε⁻¹),(2:4)); # (-1/(π)) .*
    kcross_c2t(e,kz,gx,gy,gz)
end

function M(Hin::AbstractArray{T,1},ε⁻¹,kz,gx,gy,gz)::AbstractArray{T,1} where T
    Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
    HinA = reshape(Hin,(2,Nx,Ny,Nz))
    HoutA = M(HinA,ε⁻¹,kz,gx,gy,gz)
    return vec(HoutA)
end

# M̂(ε⁻¹,kz,gx,gy,gz) = LinearMap{T}(H -> M(H::AbstractArray{T,1},ε⁻¹,kz,gx,gy,gz),(2*length(gx)*length(gy)*length(gz)),ishermitian=true,ismutating=false)
# M̂(ε⁻¹,kz,gx,gy,gz) = LinearMap{T}(H -> M(H::AbstractArray{T,1},ε⁻¹,kz,gx,gy,gz),(2*length(gx)*length(gy)*length(gz)),ishermitian=true,ismutating=false) where T
M̂(ε⁻¹,kz,gx,gy,gz) = LinearMap(H -> M(H,ε⁻¹,kz,gx,gy,gz),(2*length(gx)*length(gy)*length(gz)),ishermitian=true,ismutating=false)


function Mₖ(Hin::AbstractArray{T,1},ε⁻¹,kz,gx,gy,gz)::AbstractArray{T,1} where T
    Nx,Ny,Nz = size(ε⁻¹)[end-2:end]
    HinA = reshape(Hin,(2,Nx,Ny,Nz))
    HoutA = Mₖ(HinA,ε⁻¹,kz,gx,gy,gz)
    return vec(HoutA)
end

# M̂ₖ(ε⁻¹,kz,gx,gy,gz) = LinearMap{T}(H -> Mₖ(H::AbstractArray{T,1},ε⁻¹,kz,gx,gy,gz),(2*length(gx)*length(gy)*length(gz)),ishermitian=true,ismutating=false) where T
M̂ₖ(ε⁻¹,kz,gx,gy,gz) = LinearMap(H -> Mₖ(H,ε⁻¹,kz,gx,gy,gz),(2*length(gx)*length(gy)*length(gz)),ishermitian=true,ismutating=false)
# M̂ₖ(ε⁻¹,kz,gx,gy,gz) = LinearMap{T}(H -> Mₖ(H::AbstractArray{T,1},ε⁻¹,kz,gx,gy,gz),(2*length(gx)*length(gy)*length(gz)),ishermitian=true,ismutating=false)

#

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
		kpg, mag, m, n = Zygote.forwarddiff(kz) do kz
                            zyg_KplusG(kz,gx[i],gy[j],gz[k])
                        end
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
		kpg, mag, m, n = Zygote.forwarddiff(kz) do kz
                            zyg_KplusG(kz,gx[i],gy[j],gz[k])
                        end
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

function zyg_kcrossinv_t2c(Hin,kz,gx,gy,gz)
    # Hout = Array{ComplexF64}(undef,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
	@inbounds for i=1:Nx,j=1:Ny,k=1:Nz
		kpg, mag, m, n = Zygote.forwarddiff(kz) do kz
                            zyg_KplusG(kz,gx[i],gy[j],gz[k])
                        end
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
		kpg, mag, m, n = Zygote.forwarddiff(kz) do kz
                            zyg_KplusG(kz,gx[i],gy[j],gz[k])
                        end
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

# old non-mutating operator methods using StaticArrays

function t2c(Hin::AbstractArray{ComplexF64,4},kpG::AbstractArray{KVec,3})::AbstractArray{ComplexF64,4}
    # Hout = Array{ComplexF64}(undef,(3,size(Hin)[2:end]...))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds scale = ds.kpG[i,j,k].mag
        @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].m[1] + Hin[2,i,j,k] * kpG[i,j,k].n[1] ) * scale
        @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].m[2] + Hin[2,i,j,k] * kpG[i,j,k].n[2] ) * scale
        @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].m[3] + Hin[2,i,j,k] * kpG[i,j,k].n[3] ) * scale
    end
    return copy(Hout)
end

t2c(Hin::AbstractArray{ComplexF64,4},ds::MaxwellData)::AbstractArray{ComplexF64,4} = t2c(Hin,ds.kpG)


function c2t(Hin::AbstractArray{ComplexF64,4},kpG::AbstractArray{KVec,3})::AbstractArray{ComplexF64,4}
    # Hout = Array{ComplexF64}(undef,(2,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,2,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds Hout[1,i,j,k] =  Hin[1,i,j,k] * kpG[i,j,k].m[1] + Hin[2,i,j,k] * kpG[i,j,k].m[2] + Hin[3,i,j,k] * kpG[i,j,k].m[3]
        @inbounds Hout[2,i,j,k] =  Hin[1,i,j,k] * kpG[i,j,k].n[1] + Hin[2,i,j,k] * kpG[i,j,k].n[2] + Hin[3,i,j,k] * kpG[i,j,k].n[3]
    end
    return copy(Hout)
end

function zcross_t2c(Hin::AbstractArray{ComplexF64,4},kpG::AbstractArray{KVec,3})::AbstractArray{ComplexF64,4}
    # Hout = zeros(ComplexF64,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds Hout[1,i,j,k] = -Hin[1,i,j,k] * kpG[i,j,k].m[2] - Hin[2,i,j,k] * kpG[i,j,k].n[2]
        @inbounds Hout[2,i,j,k] =  Hin[1,i,j,k] * kpG[i,j,k].m[1] + Hin[2,i,j,k] * kpG[i,j,k].n[1]
        @inbounds Hout[3,i,j,k] = 0.0
    end
    return copy(Hout)
end

function kcross_t2c(Hin::AbstractArray{ComplexF64,4},kpG::AbstractArray{KVec,3})::AbstractArray{ComplexF64,4}
    # Hout = Array{ComplexF64}(undef,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds scale = -kpG[i,j,k].mag
        @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].n[1] - Hin[2,i,j,k] * kpG[i,j,k].m[1] ) * scale
        @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].n[2] - Hin[2,i,j,k] * kpG[i,j,k].m[2] ) * scale
        @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].n[3] - Hin[2,i,j,k] * kpG[i,j,k].m[3] ) * scale
    end
    return copy(Hout)
end

function kcross_c2t(Hin::AbstractArray{ComplexF64,4},kpG::AbstractArray{KVec,3})::AbstractArray{ComplexF64,4}
    # Hout = Array{ComplexF64}(undef,(2,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,2,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds scale = kpG[i,j,k].mag
        @inbounds at1 = Hin[1,i,j,k] * kpG[i,j,k].m[1] + Hin[2,i,j,k] * kpG[i,j,k].m[2] + Hin[3,i,j,k] * kpG[i,j,k].m[3]
        @inbounds at2 = Hin[1,i,j,k] * kpG[i,j,k].n[1] + Hin[2,i,j,k] * kpG[i,j,k].n[2] + Hin[3,i,j,k] * kpG[i,j,k].n[3]
        @inbounds Hout[1,i,j,k] =  -at2 * scale
        @inbounds Hout[2,i,j,k] =  at1 * scale
    end
    return copy(Hout)
end

function kcrossinv_t2c(Hin::AbstractArray{ComplexF64,4},kpG::AbstractArray{KVec,3})::AbstractArray{ComplexF64,4}
    # Hout = Array{ComplexF64}(undef,(3,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds scale = 1 / kpG[i,j,k].mag
        @inbounds Hout[1,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].n[1] - Hin[2,i,j,k] * kpG[i,j,k].m[1] ) * scale
        @inbounds Hout[2,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].n[2] - Hin[2,i,j,k] * kpG[i,j,k].m[2] ) * scale
        @inbounds Hout[3,i,j,k] = ( Hin[1,i,j,k] * kpG[i,j,k].n[3] - Hin[2,i,j,k] * kpG[i,j,k].m[3] ) * scale
    end
    return copy(Hout)
end

function kcrossinv_c2t(Hin::AbstractArray{ComplexF64,4},kpG::AbstractArray{KVec,3})::AbstractArray{ComplexF64,4}
    # Hout = Array{ComplexF64}(undef,(2,Nx,Ny,Nz))
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,2,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds scale = -1 / kpG[i,j,k].mag
        @inbounds at1 = Hin[1,i,j,k] * kpG[i,j,k].m[1] + Hin[2,i,j,k] * kpG[i,j,k].m[2] + Hin[3,i,j,k] * kpG[i,j,k].m[3]
        @inbounds at2 = Hin[1,i,j,k] * kpG[i,j,k].n[1] + Hin[2,i,j,k] * kpG[i,j,k].n[2] + Hin[3,i,j,k] * kpG[i,j,k].n[3]
        @inbounds Hout[1,i,j,k] =  -at2 * scale
        @inbounds Hout[2,i,j,k] =  at1 * scale
    end
    return copy(Hout)
end

function ε⁻¹_dot(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3})::AbstractArray{ComplexF64,4}
    # Hout = similar(Hin)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds Hout[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,1]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,1]*Hin[3,i,j,k]
        @inbounds Hout[2,i,j,k] =  ε⁻¹[i,j,k][1,2]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[3,i,j,k]
        @inbounds Hout[3,i,j,k] =  ε⁻¹[i,j,k][1,3]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
        # @inbounds Hout[1,i,j,k] =  ε⁻¹[i,j,k][1,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][1,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][1,3]*Hin[3,i,j,k]
        # @inbounds Hout[2,i,j,k] =  ε⁻¹[i,j,k][2,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][2,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][2,3]*Hin[3,i,j,k]
        # @inbounds Hout[3,i,j,k] =  ε⁻¹[i,j,k][3,1]*Hin[1,i,j,k] + ε⁻¹[i,j,k][3,2]*Hin[2,i,j,k] + ε⁻¹[i,j,k][3,3]*Hin[3,i,j,k]
    end
    return copy(Hout)
end

function ε_dot_approx(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3})::AbstractArray{ComplexF64,4}
    # Hout = similar(Hin)
    Nx,Ny,Nz = size(Hin)[2:4]
    Hout = Zygote.Buffer(Hin,3,Nx,Ny,Nz)
    for i=1:Nx,j=1:Ny,k=1:Nz
        @inbounds ε_ave = 3 / tr(ε⁻¹[i,j,k])
        @inbounds Hout[1,i,j,k] =  ε_ave * Hin[1,i,j,k]
        @inbounds Hout[2,i,j,k] =  ε_ave * Hin[2,i,j,k]
        @inbounds Hout[3,i,j,k] =  ε_ave * Hin[3,i,j,k]
    end
    return copy(Hout)
end

function M(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3};ds::MaxwellData)::Array{ComplexF64,4}
    d = ds.𝓕 * kcross_t2c(Hin,ds.kpG);
    e = ε⁻¹_dot(d,ε⁻¹); # (-1/(π)) .*
    kcross_c2t(ds.𝓕⁻¹ * e,ds.kpG)
end

function M(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3};ds::MaxwellData)::Array{ComplexF64,1}
    HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    HoutA = M(HinA,ε⁻¹;ds)
    return vec(HoutA)
end

M̂(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3};ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> M(H,ε⁻¹;ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)

function P(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
    e = ds.𝓕⁻¹ * kcrossinv_t2c(Hin,ds);
    d = ε_dot_approx(e,ε⁻¹,ds); # (-1/(π)) .*
    kcrossinv_c2t(ds.𝓕 * d,ds)
end

function P(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
    HinA = reshape(Hin,(2,ds.Nx,ds.Ny,ds.Nz))
    HoutA = P(HinA,ε⁻¹,ds)
    return vec(HoutA)
end

P̂(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> P(H,ε⁻¹,ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)

function Mₖ(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},kpG::AbstractArray{KVec,3})::Array{ComplexF64,4}
    d = fft(zcross_t2c(Hin,kpG),(2:4));
    # e = ε⁻¹_dot(d,ε⁻¹);
    # kcross_c2t(ifft(e,(2:4)),kpG)
    e = ifft(ε⁻¹_dot(d,ε⁻¹),(2:4));
    kcross_c2t(e,kpG)
end

function Mₖ(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},k::Float64,g::MaxwellGrid)::Array{ComplexF64,4}
    Mₖ(Hin::AbstractArray{ComplexF64,4},ε⁻¹,kpG(SVector(0.,0.,k),Zygote.dropgrad(g)))
end

function Mₖ(Hin::AbstractArray{ComplexF64,4},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,4}
    Mₖ(Hin::AbstractArray{ComplexF64,4},ε⁻¹,ds.kpG)
end

function Mₖ(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},kpG::AbstractArray{KVec,3})::Array{ComplexF64,1}
    Nx,Ny,Nz = size(ε⁻¹)
    HinA = reshape(Hin,(2,Nx,Ny,Nz))
    HoutA = Mₖ(HinA,ε⁻¹,kpG)
    return -vec(HoutA)
end

function Mₖ(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},k::Float64,g::MaxwellGrid)::Array{ComplexF64,1}
    Nx,Ny,Nz = size(ε⁻¹)
    HinA = reshape(Hin,(2,Nx,Ny,Nz))
    return -vec(Mₖ(HinA,ε⁻¹,kpG(SVector(0.,0.,k),Zygote.dropgrad(g))))
end

function Mₖ(Hin::AbstractArray{ComplexF64,1},ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData)::Array{ComplexF64,1}
    Mₖ(Hin::AbstractArray{ComplexF64,1},ε⁻¹,ds.kpG)
end

M̂ₖ(ε⁻¹::Array{SHermitianCompact{3,Float64,6},3},ds::MaxwellData) = LinearMap{ComplexF64}(H::AbstractArray{ComplexF64,1} -> Mₖ(H,ε⁻¹,ds)::AbstractArray{ComplexF64,1},(2*ds.Nx*ds.Ny*ds.Nz),ishermitian=true,ismutating=false)




#########################
#
#   old stuff
#
#########################


# """
#     t2c: v (transverse vector) → a (cartesian vector)
# """
# function t2c(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return v[1] * k.m + v[2] * k.n
# end


# """
#     c2t: a (cartesian vector) → v (transverse vector)
# """
# function c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     v0 = a ⋅ k.m
#     v1 = a ⋅ k.n
#     return SVector(v0,v1)
# end

# """
#     kcross_t2c: a (cartesian vector) = k × v (transverse vector)
# """
# function kcross_t2c(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return ( v[1] * k.n - v[2] * k.m ) * k.mag
# end

# """
#     kcross_t2c!: a (cartesian vector) = k × v (transverse vector)
# """
# function kcross_t2c!(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return v = ( v[1] * k.n - v[2] * k.m ) * k.mag
# end



# """
#     kcross_c2t: v (transverse vector) = k × a (cartesian vector)
# """
# function kcross_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     at1 = a ⋅ k.m
#     at2 = a ⋅ k.n
#     v0 = -at2 * k.mag
#     v1 = at1 * k.mag
#     return SVector(v0,v1,0.0)
# end


# """
#     kcrossinv_t2c: compute a⃗ (cartestion vector) st. v⃗ (cartesian vector from two trans. vector components) ≈ k⃗ × a⃗
#     This neglects the component of a⃗ parallel to k⃗ (not available by inverting this cross product)
# """
# function kcrossinv_t2c(v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return ( v[1] * k.n - v[2] * k.m ) * ( -1 / k.mag )
# end

# """
#     kcrossinv_c2t: compute  v⃗ (transverse 2-vector) st. a⃗ (cartestion 3-vector) = k⃗ × v⃗
#     This cross product inversion is exact because v⃗ is transverse (perp.) to k⃗
# """
# function kcrossinv_c2t(a::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     at1 = a ⋅ k.m
#     at2 = a ⋅ k.n
#     v0 = -at2 * (-1 / k.mag )
#     v1 = at1 * ( -1 / k.mag )
#     return SVector(v0,v1,0.0)
# end

# """
#     ucross_t2c: a (cartesian vector) = u × v (transverse vector)
# """
# function ucross_t2c(u::SVector{3,ComplexF64},v::SVector{3,ComplexF64},k::KVec)::SVector{3,ComplexF64}
#     return cross(u,t2c(v,k))
# end

# """
#     d_from_H(H,k): cartesian position space d vector field from transverse, PW basis H vector field
# """
# function d_from_H(H::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # d_recip = [ kcross_t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     # temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( d_recip , (1,ds.Nx,ds.Ny) )), (2,3))
#     # return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
#     return reshape(reinterpret(SVector{3,ComplexF64}, (-1/(2π)) .* ( ds.𝓕 * reinterpret( ComplexF64, reshape( kcross_t2c.(H,ds.kpG), (1,ds.Nx,ds.Ny) )) ) ),(ds.Nx,ds.Ny))
# end


# """
#     e_from_d(d,ε⁻¹): e-field from d-field in cartesian position space, from division by local ε tensor
# """
# function e_from_d(d::Array{SVector{3,ComplexF64},3},ε⁻¹::Array{SHM3,2})::Array{SVector{3,ComplexF64},3}
#     return ε⁻¹ .* d
# end


# """
#     H_from_e(e,k): reciprocal space (transverse basis) H vector field from position space cartesian basis e vector field
# """
# function H_from_e(e::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # temp = (1/(2π)) .* conj.(ifft(reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))), (2,3)))
#     # e_recip = reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
#     # return [ kcross_c2t(e_recip[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     return kcross_c2t.( reshape(reinterpret(SVector{3,ComplexF64}, (1/(2π)) .* conj.( ds.𝓕⁻¹ * reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))) ) ) ,(ds.Nx,ds.Ny)) , ds.kpG)
# end

# """
#     d_from_e(d,ε): e-field from d-field in cartesian position space, from division by local ε tensor
# """
# function d_from_e(d::Array{SVector{3,ComplexF64},3},ε::Array{SHM3,2})::Array{SVector{3,ComplexF64},3}
#     return ε .* d
# end

# """
#     h_from_H(e,k):  position space cartesian basis h vector field from reciprocal space (transverse basis) H vector field
# """
# function h_from_H(H,ds::MaxwellData)
#     h_recip = [ t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     temp =   fft( reinterpret( ComplexF64, reshape( h_recip , (1,ds.Nx,ds.Ny) )), (2,3))
#     return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
# end

# function flatten(H::Array{SVector{3,ComplexF64},3})
#     # return reinterpret(ComplexF64,vec(H))
#     return reinterpret(ComplexF64,vec(permutedims(H,[2,1])))
# end

# function unflatten(Hvec,ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # return reshape(reinterpret(SVector{3,ComplexF64},Hvec),(ds.Nx,ds.Ny))
#     return permutedims(reshape(reinterpret(SVector{3,ComplexF64},Hvec),(ds.Ny,ds.Nx)), [2,1])
# end


# Define Maxwell operator function and LinearMap instantiator objects

# function M!(H::Array{SVector{3,ComplexF64}},ε⁻¹::Array{SHM3,2},ds::MaxwellData)::Array{SVector{3,ComplexF64}}
#     H .= kcross_t2c.(H,k);
#     ds.𝓕 * H;
#     H .= ε⁻¹ .* H;
#     ds.𝓕⁻¹ * H;
#     H .= kcross_c2t.(H,k)
# end

# function M!(Hv::Vector{SVector{3,ComplexF64}},ε⁻¹::Array{SHM3,2},ds::MaxwellData)::Vector{SVector{3,ComplexF64}}
#     H = reshape(Hv,size(k))
#     H .= kcross_t2c.(H,k);
#     ds.𝓕 * H;
#     H .= ε⁻¹ .* H;
#     ds.𝓕⁻¹ * H;
#     H .= kcross_c2t.(H,k)
#     return Hv
# end

# function M!(Hv::Vector{ComplexF64},ε⁻¹::Array{SHM3,2},ds::MaxwellData)::Vector{ComplexF64}
#     HSv = copy(reinterpret(SVector{3,ComplexF64}, Hv))
#     M!(HSv,ε⁻¹,ds);
#     Hv .= copy(reinterpret(ComplexF64, HSv))
# end

# function M!(Hv::Vector{ComplexF64},ε⁻¹::Array{SHM3,2},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::Vector{ComplexF64}
#     copyto!(Hw, reinterpret(SVector{3,ComplexF64},reshape(Hv,(3*size(k)[1],size(k)[2:end]...))) )
#     M!(Hw,ε⁻¹,ds);
#     copyto!(Hv, vec( reinterpret(ComplexF64,Hw) ) )
#     return Hv
# end

# function M̂ₖ!(ε⁻¹::Array{SHM3,3},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::LinearMaps.FunctionMap{ComplexF64}
#     N = 3 * ds.Nx * ds.Ny * ds.Nz
#     f = H::Vector{ComplexF64} -> M!(H,ε⁻¹,ds,Hw)
#     return LinearMap{ComplexF64}(f,f,N,N,ishermitian=true,ismutating=true)
# end

# function M̂ₖ(ε⁻¹::Array{SHM3,2},ds::MaxwellData)::LinearMaps.FunctionMap{ComplexF64}
#     N = 2 * ds.Nx * ds.Ny * ds.Nz
#     f = H -> flatten(H_from_e( e_from_d( d_from_H( unflatten(H,ds), ds ), ε⁻¹ ), ds))
#     fc = H -> flatten(H_from_e( d_from_e( d_from_H( unflatten(H,ds), ds ), SHM3.(inv.(ε⁻¹)) ), ds))
#     return LinearMap{ComplexF64}(f,fc,N,N,ishermitian=true)
# end

# function z_cross_H(H::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     z_cross_h_recip = [ ucross_t2c(SVector{3,ComplexF64}(0.,0.,1.),H[i,j,k],ds.kpG[i,j,k]) for i=1:ds.Nx, j=1:ds.Ny, k=1:ds.Nz]
#     temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( z_cross_h_recip , (1,ds.Nx,ds.Ny,ds.Nz) )), (2,3))
#     return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny,ds.Nz))
# end

# function ∂ₖM̂ₖ(ε⁻¹::Array{SHM3,3},ds::MaxwellData)::LinearMaps.FunctionMap{ComplexF64}
#     N = 2 * ds.Nx * ds.Ny * ds.Nz
#     f = H -> flatten(H_from_e( e_from_d( z_cross_H( unflatten(H,ds), ds ), ε⁻¹ ), ds))
#     return LinearMap{ComplexF64}(f,f,N,N,ishermitian=true)
# end


# Define approximate inversion operator P̂ₖ ≈ M̂ₖ⁻¹ to use as preconditioner


# function P!(H::Array{SVector{3,ComplexF64}},ε::Array{SHM3,3},ds::MaxwellData)::Array{SVector{3,ComplexF64}}
#     H .= kcrossinv_t2c.(H,k);
#     ds.𝓕⁻¹ * H;
#     H .= ε .* H;
#     ds.𝓕 * H;
#     H .= kcrossinv_c2t.(H,k)
# end

# function P!(Hv::Vector{ComplexF64},ε::Array{SHM3,3},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::Vector{ComplexF64}
#     copyto!(Hw, reinterpret(SVector{3,ComplexF64},reshape(Hv,(3*size(k)[1],size(k)[2:end]...))) )
#     P!(Hw,ε,ds);
#     copyto!(Hv, vec( reinterpret(ComplexF64,Hw) ) )
#     return Hv
# end

# function P̂ₖ!(ε::Array{SHM3,3},ds::MaxwellData,Hw::Array{SVector{3,ComplexF64}})::LinearMaps.FunctionMap{ComplexF64}
#     N = 3 * ds.Nx * ds.Ny * ds.Nz
#     f = H::Vector{ComplexF64} -> P!(H,ε,ds,Hw)
#     return LinearMap{ComplexF64}(f,f,N,N,ismutating=true)
# end


# """
#     e_from_H_approx(H,k): cartesian position space d vector field from transverse, PW basis H vector field
# """
# function e_from_H_approx(H::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # d_recip = [ kcross_t2c(H[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     # temp =  (-1/(2π)) .* fft( reinterpret( ComplexF64, reshape( d_recip , (1,ds.Nx,ds.Ny) )), (2,3))
#     # return reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
#     return reshape(reinterpret(SVector{3,ComplexF64}, (-1/(2π)) .* ( ds.𝓕 * reinterpret( ComplexF64, reshape( kcrossinv_t2c.(H,ds.kpG), (1,ds.Nx,ds.Ny,ds.Nz) )) ) ),(ds.Nx,ds.Ny,ds.Nz))
# end

# """
#     H_from_d(e,k): reciprocal space (transverse basis) H vector field from position space cartesian basis e vector field
# """
# function H_from_d(e::Array{SVector{3,ComplexF64},3},ds::MaxwellData)::Array{SVector{3,ComplexF64},3}
#     # temp = (1/(2π)) .* conj.(ifft(reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny))), (2,3)))
#     # e_recip = reshape(reinterpret(SVector{3,ComplexF64},temp),(ds.Nx,ds.Ny))
#     # return [ kcross_c2t(e_recip[i,j],ds.kpG[i,j]) for i=1:ds.Nx, j=1:ds.Ny]
#     return kcrossinv_c2t.( reshape(reinterpret(SVector{3,ComplexF64}, (1/(2π)) .* conj.( ds.𝓕⁻¹ * reinterpret(ComplexF64,reshape(e,(1,ds.Nx,ds.Ny,ds.Nz))) ) ) ,(ds.Nx,ds.Ny,ds.Nz)) , ds.kpG)
# end

# """
# approximate inversion operator P̂ₖ ≈ M̂ₖ⁻¹ to use as preconditioner
# """
# function P̂ₖ(ε::Array{SHM3,3},ds::MaxwellData)::LinearMaps.FunctionMap
#     N = 2 * ds.Nx * ds.Ny * ds.Nz
#     f = H -> flatten(H_from_d( d_from_e( e_from_H_approx( unflatten(H,ds), ds ), ε ), ds))
#     fc = H -> flatten(H_from_d( e_from_d( e_from_H_approx( unflatten(H,ds), ds ), SHM3.(inv.(ε)) ), ds))
#     return LinearMap{ComplexF64}(f,fc,N,N)
# end
