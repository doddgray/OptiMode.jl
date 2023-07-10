export E_relpower_xyz, Eslices, count_E_nodes, windowed_E_slices, mode_viable, mode_idx, Eperp_max, 𝓐

"""
Calculate the relative E-field power along each grid axis (integrated over space) for a mode field, and return these values as a 3-Tuple.
This is useful for categorizing/distinguishing mode polarization, for example quasi-TE modes might give values like (0.95,0.04,0.01) while
quasi-TM modes give values more like (0.01,0.98,0.01).
"""
function E_relpower_xyz(eps::AbstractArray{T,4},E::AbstractArray{Complex{T},3}) where T<:Real
    return normalize( real( @tullio E_abspower_xyz[i] := conj(E)[i,ix,iy] * eps[i,j,ix,iy] * E[j,ix,iy] ))
end

function E_relpower_xyz(eps::AbstractArray{T,5},E::AbstractArray{Complex{T},4}) where T<:Real
    return normalize( real( @tullio E_abspower_xyz[i] := conj(E)[i,ix,iy,iz] * eps[i,j,ix,iy,iz] * E[j,ix,iy,iz] ))
end

"""
Return slices (views) of E-field component `component_idx` along each axis intersecting the point at Cartesian index `pos_idx`
"""
function Eslices(E,component_idx,pos_idx::CartesianIndex{2})
    return ( view(E,component_idx,:,pos_idx[2]), view(E,component_idx,pos_idx[1],:) ) 
end

"""
Return slices (views) of E-field component `component_idx` along each axis intersecting the point at Cartesian index `pos_idx`
"""
function Eslices(E,component_idx,pos_idx::CartesianIndex{3})
    return ( view(E,component_idx,:,pos_idx[2],pos_idx[3]), view(E,component_idx,pos_idx[1],:,pos_idx[3]), view(E,component_idx,pos_idx[1],pos_idx[2],:) ) 
end

"""
Count the number of E-field zero crossings along each grid axis intersecting the peak E-field intensity maximum for a given field component `component_idx`.
This assumes that the input E-field amplitude is normalized to and real at (phase=0) the peak amplitude component-wise amplitude. 
Slices of the specified E-field component are cropped at the points where the component magnitude drops below `rel_amp_min` to avoid counting 
zero-crossings in numerical noise where the E-field magnitude is negligible.

This is useful when filtering for a specific mode order (eg. TE₀₀ or TM₂₁) in the presence of mode-crossings.
"""
function count_E_nodes(E,eps,component_idx;rel_amp_min=0.1) 
    peak_idx = argmax(real(_3dot(E,eps,E)[component_idx,..]))
    E_slcs = Eslices(real(E),component_idx,peak_idx)
    node_counts = map(E_slcs) do E_slice
        min_idx = findfirst(x->(x>rel_amp_min),abs.(E_slice))
        max_idx = findlast(x->(x>rel_amp_min),abs.(E_slice))
        n_zero_xing = sum(abs.(diff(sign.(E_slice[min_idx:max_idx]))))
        return n_zero_xing
    end
    return node_counts
end

"""
Utility function for debugging count_E_nodes function.
inspect output with something like: 
    slcs_inds = windowed_E_slices(E,eps,component_idx;rel_amp_min=0.1)
    ind_min_xslc, ind_max_xslc = slcs_inds[1][2:3]
    E_xslc = slcs_inds[1][1]
    scatterlines(x(grid)[ind_min_xslc:ind_max_xslc],real(E_xslc[ind_min_xslc:ind_max_xslc]))
"""
function windowed_E_slices(E,eps,component_idx;rel_amp_min=0.1) 
    peak_idx = argmax(real(_3dot(E,eps,E)[component_idx,..]))
    E_slcs = Eslices(real(E),component_idx,peak_idx)
    slices_and_window_inds = map(E_slcs) do E_slice
        min_idx = findfirst(x->(x>rel_amp_min),abs.(E_slice))
        max_idx = findlast(x->(x>rel_amp_min),abs.(E_slice))
        # n_zero_xing = sum(abs.(diff(sign.(E_slice[min_idx:max_idx]))))
        return E_slice, min_idx, max_idx
    end
    return slices_and_window_inds
end


"""
Return `true` if the following conditions are met:
    (1) The mode E-field `E` is dominantly polarized along axis index `pol_idx`
    (2) The Hermite-Gaussian mode order computed by `count_E_nodes` is equal to `mode_order`
Otherwise, return `false`.  
"""
function mode_viable(E,eps;pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
    Epwr = E_relpower_xyz(eps,E)
    Epol_axind = argmax(Epwr)
    E_mode_order = count_E_nodes(E,eps,Epol_axind;rel_amp_min)
    return ( isequal(argmax(Epwr),pol_idx) && isequal(E_mode_order,mode_order) )
end

"""
Identify the mode of a given order/polarization in an vector of
eigenmode E-fields `Es`, length `nbands`, found by solving Helmholtz Equation at a single frequency.
"""
function mode_idx(Es::AbstractVector,eps;pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
    return findfirst(EE->mode_viable(EE,eps;pol_idx,mode_order,rel_amp_min),Es)
end

"""
Identify the mode of a given order/polarization for each frequency index in an vector of
eigenmode E-fields `Es`, length `nbands`.
"""
function mode_idx(Es,eps::AbstractVector;pol_idx=1,mode_order=(0,0),rel_amp_min=0.4)
    n_freq = first(size(Es))
    return [ findfirst(EE->mode_viable(EE,eps[fidx];pol_idx,mode_order,rel_amp_min),Es[fidx,:]) for fidx=1:n_freq ]
end

"""
Find the maximum-magnitude complex E-field value along axes 1 and 2.
Normalizing mode E-fields by this quantity makes their maximum amplitude transverse to the optical axis real and equal to one.
This is convenient for plotting and perturbation theory/nonlinear coupling calculations.   
"""
function Eperp_max(E::AbstractArray{T,3}) where T
    Eperp = view(E,1:2,:,:)
    maximum(abs,Eperp)
end

function Eperp_max(E::AbstractArray{T,4}) where T
    Eperp = view(E,1:2,:,:,:)
    maximum(abs,Eperp)
end

"""
Calculate the effective area `𝓐` of a mode-field `E` with modal effective index `n` and modal group index `ng`
"""
𝓐(n,ng,E) = inv( n * ng * Eperp_max(E)^2)

const χ⁽²⁾_default = SArray{Tuple{3,3,3}}(zeros(Float64,(3,3,3)))

"""
`proc_modes_shg` processes a set of mode-solutions `kk_ev` (`n_bands` k-eigenvector pairs at each frequency) over a range of frequencies `oms` 
and the second harmonics of those frequencies, `2.*oms`. Effective and group indices, phase mismatches and SHG coupling coefficients are calculated
as a function of frequency and returned.

TODO: the nonlinear sucsceptibility and coupling calculation is somewhat hardcoded with variable names referring to Lithium Niobate. Fix names and provide
kwarg inputs to specify the coupling of interest.
"""
function proc_modes_shg(oms,pp,kk_ev;pol_idx_fund=1,mode_order_fund=(0,0),pol_idx_sh=1,mode_order_sh=(0,0),rel_amp_min=0.4,num_bands=size(first(first(kk_ev)))[2])
    ks, evecs = kk_ev
    
    eps =  [copy(smooth(oo,pp,:fεs,false,geom_fn,grid).data) for oo in oms]
    epsi =  [copy(smooth(oo,pp,:fεs,true,geom_fn,grid).data) for oo in oms]
    deps_dom = [ ForwardDiff.derivative(oo->copy(getproperty(smooth(oo,pp,(:fεs,:fεs),[false,true],geom_fn,grid,kottke_smoothing)[1],:data)),om) for om in oms ]
    mags_mns = [mag_mn(kk,grid) for kk in ks]
    Es = [-1im * ε⁻¹_dot(fft(kx_tc(evecs[omidx,bndidx],mags_mns[omidx,bndidx][2],mags_mns[omidx,bndidx][1]),(2:3)),epsi[omidx]) for omidx=1:length(oms), bndidx=1:num_bands]
    Enorms =  [ EE[argmax(abs2.(EE))] for EE in Es ]
    Es = Es ./ Enorms
    
    ngs = [ group_index(ks[fidx,bidx],evecs[fidx,bidx],oms[fidx],epsi[fidx],deps_dom[fidx],grid) for fidx=1:length(oms),bidx=1:num_bands ]
    neffs = ks ./ repeat(oms,1,num_bands) 

    nω = Int(length(oms) // 2)  # oms has all first and second harmonic frequencies, nω is just the number of first harmonic frequencies

    # bnd_inds_fund   = [mode_idx(Es[omidx,:],eps[omidx];pol_idx=pol_idx_fund,mode_order=mode_order_fund,rel_amp_min) for omidx=1:nω]
    # bnd_inds_sh     = [mode_idx(Es[omidx,:],eps[omidx];pol_idx=pol_idx_sh,mode_order=mode_order_sh,rel_amp_min) for omidx=(nω+1):(2*nω)]
    bnd_inds_fund   = mode_idx(Es[1:nω,:],eps[1:nω];pol_idx=pol_idx_fund,mode_order=mode_order_fund,rel_amp_min)
    bnd_inds_sh     = mode_idx(Es[(nω+1):(2*nω),:],eps[(nω+1):(2*nω)];pol_idx=pol_idx_sh,mode_order=mode_order_sh,rel_amp_min)
    bnd_inds        = vcat(bnd_inds_fund,bnd_inds_sh)

    neffs_fund  =    [ neffs[idx,bnd_inds[idx]] for idx=1:nω]
    ngs_fund    =    [ ngs[idx,bnd_inds[idx]] for idx=1:nω]
    Es_fund     =    [ Es[idx,bnd_inds[idx]] for idx=1:nω]
    eps_fund    =    @view eps[1:nω]

    neffs_sh   =    [ neffs[idx,bnd_inds[idx]] for idx=(nω+1):(2*nω) ]
    ngs_sh     =    [ ngs[idx,bnd_inds[idx]] for idx=(nω+1):(2*nω) ]
    Es_sh      =    [ Es[idx,bnd_inds[idx]] for idx=(nω+1):(2*nω) ]
    eps_sh     =    @view eps[(nω+1):(2*nω)]
    
    λs_sh      =    inv.(ωs_in[(nω+1):(2*nω)])
    Λs         =    λs_sh ./ ( neffs_sh .- neffs_fund )

    As_fund = 𝓐.(neffs_fund, ngs_fund, Es_fund)
    As_sh = 𝓐.(neffs_sh, ngs_sh, Es_sh)
    Ês_fund = [Es_fund[i] * sqrt(As_fund[i] * neffs_fund[i] * ngs_fund[i]) for i=1:length(Es_fund)]
    Ês_sh = [Es_sh[i] * sqrt(As_sh[i] * neffs_sh[i] * ngs_sh[i]) for i=1:length(Es_sh)]
    𝓐₁₂₃ = ( As_sh .* As_fund.^2  ).^(1.0/3.0)
    χ⁽²⁾ = -1 .* smooth([(omm,omm,2omm) for omm in ωs_in[1:length(Es_fund)]],ps[1],:fχ⁽²⁾s,false,χ⁽²⁾_default,geom_fn,grid,volfrac_smoothing)
    χ⁽²⁾_rel = [abs.(chi2) / maximum(abs.(chi2)) for chi2 in χ⁽²⁾]
    χ⁽²⁾xxx	=	[ view(chi2,1,1,1,:,:)  for chi2 in χ⁽²⁾]
    χ⁽²⁾xxx_LN	=	[ chi2xxx[argmax(abs.(chi2xxx))] for chi2xxx in χ⁽²⁾xxx]
    # χ⁽²⁾xxx_rel	=	[ chi2xxx ./ chi2xxx_LN  for (chi2xxx,chi2xxx_LN) in zip(χ⁽²⁾xxx,χ⁽²⁾xxx_LN)]

    𝓞 = [ real( sum( dot( conj.(Ês_sh[ind]), _dot(χ⁽²⁾_rel[ind],Ês_fund[ind],Ês_fund[ind]) ) ) ) / 𝓐₁₂₃[ind] * δ(grid) for ind=1:length(Es_sh)] #
    # 𝓞_rel = 𝓞/maximum(𝓞)

    χ⁽²⁾xxx_LN_cmV⁻¹ = χ⁽²⁾xxx_LN .* 1e-10
    deff_cmV⁻¹ = χ⁽²⁾xxx_LN_cmV⁻¹.* (2/π)
    𝓐₁₂₃_cm² = 𝓐₁₂₃ .* 1e-8
    c₀_cm = 3e10
    ε₀_Fcm⁻¹ = 8.854e-14 # F/cm, ε₀	= 8.854e-12 F/m
    λ_cm = inv.(ωs_fund) .* 1e-4
    om_radHz = 2π .* (c₀_cm ./ λ_cm)
    κ_sqrtW⁻¹cm⁻¹ = ( sqrt(2) * om_radHz .* deff_cmV⁻¹ .* 𝓞) ./ sqrt.( c₀_cm^3 * ε₀_Fcm⁻¹ .* neffs_sh .* neffs_fund.^2 .* 𝓐₁₂₃_cm²)
    κ²_W⁻¹cm⁻² = abs2.(κ_sqrtW⁻¹cm⁻¹)
    κ²_pctW⁻¹cm⁻² = 100.0 .* κ²_W⁻¹cm⁻²
    # calculate phase matching bandwidth based on group index mismatch
    Δng         =       ngs_sh .- ngs_fund
    Δf_Hz_cm    =       c₀_cm ./ ( 4.0 .* abs.(Δng) )   # Phase matching bandwidth around first harmonic in Hz⋅cm (or in Hz of a 1cm waveguide)
    f_Hz        =       c₀_cm ./ λ_cm                   # fundamental frequencies in Hz
    ∂λ_∂f       =       -c₀_cm ./ (f_Hz.^2)       # partial derivative of each fundatmental vacuum wavelength w.r.t. frequency
    nm_per_cm   =       1e7
    Δλ_nm_cm    =       abs.( ∂λ_∂f .*  Δf_Hz_cm .* nm_per_cm )  # Phase matching bandwidth around first harmonic in nm⋅cm (or in nm of a 1cm waveguide)

    return neffs_fund, ngs_fund, Es_fund, eps_fund, neffs_sh, ngs_sh, Es_sh, eps_sh, Λs, κ²_pctW⁻¹cm⁻², Δλ_nm_cm
end
