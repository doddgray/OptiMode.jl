export E_relpower_xyz, Eslices, count_E_nodes, windowed_E_slices, mode_viable, mode_idx, Eperp_max, 𝓐, effective_area

# alias for the effective-area function 𝓐 with an ASCII-friendly name

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
const effective_area = 𝓐

