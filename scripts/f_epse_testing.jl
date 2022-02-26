####### Testing


##
using OptiMode
mats = mats = [MgO_LiNbO₃,Si₃N₄,SiO₂,Vacuum];
n_mats = length(mats);
f_ε_mats, f_ε_mats! = _f_ε_mats(mats,(:ω,));
ε,∂ωε,∂²ωε = ε_views(f_ε_mats([1.0,]),n_mats);
εₑᵣ_∂ωεₑᵣ_herm(0.3,ε[1],ε[2],∂ωε[1],∂ωε[2]);
εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ_herm(0.3,ε[1],ε[2],∂ωε[1],∂ωε[2],∂²ωε[1],∂²ωε[2]);
S0 = normcart(normalize(rand(3)));
r0 = rand();
epse1,depse1 = εₑ_∂ωεₑ(r0,S0,ε[1],ε[2],∂ωε[1],∂ωε[2]);
epse1h,depse1h = εₑ_∂ωεₑ_herm(r0,S0,ε[1],ε[2],∂ωε[1],∂ωε[2]);
@assert epse1 ≈ epse1h
@assert depse1 ≈ depse1h
epse2,depse2,ddepse2 = εₑ_∂ωεₑ_∂²ωεₑ_herm(r0,S0,ε[1],ε[2],∂ωε[1],∂ωε[2],∂²ωε[1],∂²ωε[2]);
epse2h,depse2h,ddepse2h = εₑ_∂ωεₑ_∂²ωεₑ_herm(r0,S0,ε[1],ε[2],∂ωε[1],∂ωε[2],∂²ωε[1],∂²ωε[2]);
@assert epse2 ≈ epse2h
@assert depse2 ≈ depse2h
@assert ddepse2 ≈ ddepse2h


####

pA = @variables v1 v2 v3
npA = length(pA)
A = [ rand()*mapreduce(x->x^rand(1:8),*,pA) for i=1:30 ];
fj_A_sym, fjh_A_sym = _fj_fjh_sym(A,pA);
fA, fjA, fjhA = eval_fn_oop(A,pA), eval_fn_oop(fj_A_sym,pA), eval_fn_oop(fjh_A_sym,pA);
fA!, fjA!, fjhA! = eval_fn_ip(A,pA), eval_fn_ip(fj_A_sym,pA), eval_fn_ip(fjh_A_sym,pA);
fA_out = fA(rand(npA));
fjA_out = fjA(rand(npA));
fjhA_out = fjhA(rand(npA));
fA!(similar(fA_out),rand(npA));
fjA!(similar(fjA_out),rand(npA));
fjhA!(similar(fjhA_out),rand(npA));

fA2_ex, fA!2_ex = build_function(A,pA);
fA2, fA!2 = eval(fA2_ex), eval(fA!2_ex);
fA2_out = fA2(rand(npA));
fA!2(similar(fA2_out),rand(npA));

fjA2_ex, fjA!2_ex = build_function(fj_A_sym,pA);
fjA2, fjA!2 = eval(fjA2_ex), eval(fjA!2_ex);
fjA2_out = fjA2(rand(npA));
fjA!2(similar(fjA2_out),rand(npA));

pA0 = rand(npA);

f_epse2DH_sym, p2DH = _f_epse2DH_sym();
# fj_epse2DH_sym = _fj_sym(f_epse2DH_sym,p2DH);
fj_epse2DH_sym, fjh_epse2DH_sym = _fj_fjh_sym(f_epse2DH_sym,p2DH);
f_epse2DH   = eval_fn_oop(f_epse2DH_sym,p2DH);
fj_epse2DH  = eval_fn_oop(fj_epse2DH_sym,p2DH);
fjh_epse2DH  = eval_fn_oop(fjh_epse2DH_sym,p2DH);
f_epse2DH!   = eval_fn_ip(f_epse2DH_sym,p2DH);
fj_epse2DH!  = eval_fn_ip(fj_epse2DH_sym,p2DH);
fjh_epse2DH! = eval_fn_ip(fjh_epse2DH_sym,p2DH);
fout_2DH = f_epse2DH(rand(14));
fjout_2DH = fj_epse2DH(rand(14));
fjhout_2DH = fjh_epse2DH(rand(14));
f_epse2DH!(similar(fout_2DH),rand(14));
fj_epse2DH!(similar(fjout_2DH),rand(14));
fjh_epse2DH!(similar(fjhout_2DH),rand(14));

f_epse3DH_sym, p3DH = _f_epse3DH_sym();
# fj_epse3DH_sym = _fj_sym(f_epse3DH_sym,p3DH);
fj_epse3DH_sym, fjh_epse3DH_sym = _fj_fjh_sym(f_epse3DH_sym,p3DH);
fj_epse3DH = eval_fn_oop(fj_epse3DH_sym,p3DH);
fjh_epse3DH = eval_fn_oop(fjh_epse3DH_sym,p3DH);
f_epse3DH = eval_fn_oop(f_epse3DH_sym,p3DH);
fj_epse3DH! = eval_fn_ip(fj_epse3DH_sym,p3DH);
fjh_epse3DH! = eval_fn_ip(fjh_epse3DH_sym,p3DH);
f_epse3DH! = eval_fn_ip(f_epse3DH_sym,p3DH);
fout_3DH = f_epse3DH(rand(16));
fjout_3DH = fj_epse3DH(rand(16));
fjhout_3DH = fjh_epse3DH(rand(16));
f_epse3DH!(similar(fout_3DH),rand(16));
fj_epse3DH!(similar(fjout_3DH),rand(16));
fjh_epse3DH!(similar(fjhout_3DH),rand(16));

f_epse3D_sym, p3D = _f_epse3D_sym(); 
# fj_epse3D_sym = _fj_sym(f_epse3D_sym,p3D);
fj_epse3D_sym, fjh_epse3D_sym = _fj_fjh_sym(f_epse3D_sym,p3D);
f_epse3D = eval_fn_oop(f_epse3D_sym,p3D);
fj_epse3D = eval_fn_oop(fj_epse3D_sym,p3D);
fjh_epse3D = eval_fn_oop(fjh_epse3D_sym,p3D);
f_epse3D! = eval_fn_ip(f_epse3D_sym,p3D);
fj_epse3D! = eval_fn_ip(fj_epse3D_sym,p3D);
fjh_epse3D! = eval_fn_ip(fjh_epse3D_sym,p3D);
fout_3D = f_epse3D(rand(22));
fjout_3D = fj_epse3D(rand(22));
fjhout_3D = fjh_epse3D(rand(22));
f_epse3D!(similar(fout_3D),rand(22));
fj_epse3D!(similar(fjout_3D),rand(22));
fjh_epse3D!(similar(fjhout_3D),rand(22));

###########################

# include("f_eps_mats.jl")

# f_εₑ(x)  = f_epse3D(x)
# fj_εₑ(x) = fj_epse3D(x)
# fjh_εₑ(x) = fjh_epse3D(x)
# f_εₑ!(res,x)  = f_epse3D!(res,x)
# fj_εₑ!(res,x) = fj_epse3D!(res,x)
# fjh_εₑ!(res,x) = fjh_epse3D!(res,x)
# rand_p_εₑ() = rand(22)

# f_εₑ(x)  = f_epse2DH(x)
# fj_εₑ(x) = fj_epse2DH(x)
# fjh_εₑ(x) = fjh_epse2DH(x)
# f_εₑ!(res,x)  = f_epse2DH!(res,x)
# fj_εₑ!(res,x) = fj_epse2DH!(res,x)
# fjh_εₑ!(res,x) = fjh_epse2DH!(res,x)
# rand_p_εₑ() = rand(14)

# function ∂ωεₑ(r₁,θ::Real,ε₁::SHermitianCompact{3,T,6},ε₂::SHermitianCompact{3,T,6},∂ω_ε₁,∂ω_ε₂) where T
#     return @views @inbounds reshape( fj_εₑ(vcat(r₁,θ,getproperty(ε₁,:lowertriangle),getproperty(ε₂,:lowertriangle)))[:,2:15]  * vcat(zeros(2),vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )
# end 

# function εₑ_∂ωεₑ(r₁,θ::Real,ε₁::SHermitianCompact{3,T,6},ε₂::SHermitianCompact{3,T,6},∂ω_ε₁,∂ω_ε₂) where T
#     fj_εₑ_12 = fj_εₑ(vcat(r₁,θ,getproperty(ε₁,:lowertriangle),getproperty(ε₂,:lowertriangle)));
#     f_εₑ_12, j_εₑ_12 = @views @inbounds fj_εₑ_12[:,1], fj_εₑ_12[:,2:15];
#     εₑ_12 = @views reshape(f_εₑ_12,(3,3))
#     v_∂ω = vcat(zeros(2),getproperty(∂ω_ε₁,:lowertriangle),getproperty(∂ω_ε₂,:lowertriangle));
#     ∂ω_εₑ_12 = @views reshape( j_εₑ_12 * v_∂ω, (3,3) );
#     return εₑ_12, ∂ω_εₑ_12
# end

# function εₑ_∂ωεₑ_∂²ωεₑ(r₁,θ::Real,ε₁::SHermitianCompact{3,T,6},ε₂::SHermitianCompact{3,T,6},∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂) where T
#     fjh_εₑ_12 = fjh_εₑ(vcat(r₁,θ,getproperty(ε₁,:lowertriangle),getproperty(ε₂,:lowertriangle)));
#     f_εₑ_12, j_εₑ_12, h_εₑ_12 = @views @inbounds fjh_εₑ_12[:,1], fjh_εₑ_12[:,2:15], reshape(fjh_εₑ_12[:,16:211],(9,14,14));
#     εₑ_12 = @views reshape(f_εₑ_12,(3,3))
#     v_∂ω, v_∂²ω = vcat(zeros(2),getproperty(∂ω_ε₁,:lowertriangle),getproperty(∂ω_ε₂,:lowertriangle)), vcat(zeros(2),getproperty(∂²ω_ε₁,:lowertriangle),getproperty(∂²ω_ε₂,:lowertriangle));
#     ∂ω_εₑ_12 = @views reshape( j_εₑ_12 * v_∂ω, (3,3) );
#     ∂ω²_εₑ_12 = @views reshape( [dot(v_∂ω,h_εₑ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑ_12*v_∂²ω , (3,3) );
#     return εₑ_12, ∂ω_εₑ_12, ∂ω²_εₑ_12
# end


# fepse_out = f_εₑ(rand_p_εₑ())
# fjepse_out = fj_εₑ(rand_p_εₑ())
# fjhepse_out = fjh_εₑ(rand_p_εₑ())
# f_εₑ!(fepse_out,rand_p_εₑ())
# fj_εₑ!(fjepse_out,rand_p_εₑ())
# fjh_εₑ!(fjhepse_out,rand_p_εₑ())



∂ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)   = @views @inbounds reshape( fj_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)))[:,2:end]  * vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )

# function εₑᵣ_∂ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)
#     fj_εₑᵣ_12 = fj_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)));
#     f_εₑᵣ_12, j_εₑᵣ_12 = @views @inbounds fj_εₑᵣ_12[:,1], fj_εₑᵣ_12[:,2:end];
#     εₑᵣ_12 = @views reshape(f_εₑᵣ_12,(3,3))
#     v_∂ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂));
#     ∂ω_εₑᵣ_12 = @views reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
#     return εₑᵣ_12, ∂ω_εₑᵣ_12
# end

# function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
#     fjh_εₑᵣ_12 = fjh_εₑᵣ(vcat(r₁,vec(ε₁),vec(ε₂)));
#     f_εₑᵣ_12, j_εₑᵣ_12, h_εₑᵣ_12 = @views @inbounds fjh_εₑᵣ_12[:,1], fjh_εₑᵣ_12[:,2:20], reshape(fjh_εₑᵣ_12[:,21:381],(9,19,19));
#     εₑᵣ_12 = @views reshape(f_εₑᵣ_12,(3,3))
#     v_∂ω, v_∂²ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), vcat(0.0,vec(∂²ω_ε₁),vec(∂²ω_ε₂));
#     ∂ω_εₑᵣ_12 = @views reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
#     ∂ω²_εₑᵣ_12 = @views reshape( [dot(v_∂ω,h_εₑᵣ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑᵣ_12*v_∂²ω , (3,3) );
#     return εₑᵣ_12, ∂ω_εₑᵣ_12, ∂ω²_εₑᵣ_12
# end

function εₑᵣ_∂ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)
    fj_εₑᵣ_12 = similar(ε₁,9,20)
    fj_εₑᵣ!(fj_εₑᵣ_12,vcat(r₁,vec(ε₁),vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12 = @views @inbounds fj_εₑᵣ_12[:,1], fj_εₑᵣ_12[:,2:end];
    εₑᵣ_12 = @views reshape(f_εₑᵣ_12,(3,3))
    v_∂ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂));
    ∂ω_εₑᵣ_12 = @views reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    return εₑᵣ_12, ∂ω_εₑᵣ_12
end

function εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
    fjh_εₑᵣ_12 = similar(ε₁,9,381)
    fjh_εₑᵣ!(fjh_εₑᵣ_12,vcat(r₁,vec(ε₁),vec(ε₂)));
    f_εₑᵣ_12, j_εₑᵣ_12, h_εₑᵣ_12 = @views @inbounds fjh_εₑᵣ_12[:,1], fjh_εₑᵣ_12[:,2:20], reshape(fjh_εₑᵣ_12[:,21:381],(9,19,19));
    εₑᵣ_12 = @views reshape(f_εₑᵣ_12,(3,3))
    v_∂ω, v_∂²ω = vcat(0.0,vec(∂ω_ε₁),vec(∂ω_ε₂)), vcat(0.0,vec(∂²ω_ε₁),vec(∂²ω_ε₂));
    ∂ω_εₑᵣ_12 = @views reshape( j_εₑᵣ_12 * v_∂ω, (3,3) );
    ∂ω²_εₑᵣ_12 = @views reshape( [dot(v_∂ω,h_εₑᵣ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑᵣ_12*v_∂²ω , (3,3) );
    return εₑᵣ_12, ∂ω_εₑᵣ_12, ∂ω²_εₑᵣ_12
end

ε,∂ωε,∂²ωε = ε_views(f_ε_mats([1.0,]),n_mats);
εₑᵣ_∂ωεₑᵣ(0.3,ε[1],ε[2],∂ωε[1],∂ωε[2]);
εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(0.3,ε[1],ε[2],∂ωε[1],∂ωε[2],∂²ωε[1],∂²ωε[2]);

@inline _rotate(S,ε) = transpose(S) * (ε * S)
εₑ_∂ωεₑ(r₁,S,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂) = _rotate.((transpose(S),),εₑᵣ_∂ωεₑᵣ(r₁,_rotate(S,ε₁),_rotate(S,ε₂),_rotate(S,∂ω_ε₁),_rotate(S,∂ω_ε₂)))
εₑ_∂ωεₑ_∂²ωεₑ(r₁,S,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂) = _rotate.((transpose(S),),εₑᵣ_∂ωεₑᵣ_∂²ωεₑᵣ(r₁,_rotate(S,ε₁),_rotate(S,ε₂),_rotate(S,∂ω_ε₁),_rotate(S,∂ω_ε₂),_rotate(S,∂²ω_ε₁),_rotate(S,∂²ω_ε₂)))
εₑ_∂ωεₑ_∂²ωεₑ(r₁,S,idx1,idx2,ε,∂ω_ε,∂²ω_ε) = εₑ_∂ωεₑ_∂²ωεₑ(r₁,S,ε[idx1],ε[idx2],∂ω_ε[idx1],∂ω_ε[idx2],∂²ω_ε[idx1],∂²ω_ε[idx2])


##

∂ωεₑ(r₁,n,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)   = @views @inbounds reshape( fj_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)))[:,2:23]  * vcat(zeros(4),vec(∂ω_ε₁),vec(∂ω_ε₂)), (3,3) )

function εₑ_∂ωεₑ(r₁,n,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂)
    fj_εₑ_12 = fj_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)));
    f_εₑ_12, j_εₑ_12 = @views @inbounds fj_εₑ_12[:,1], fj_εₑ_12[:,2:23];
    εₑ_12 = @views reshape(f_εₑ_12,(3,3))
    v_∂ω = vcat(zeros(4),vec(∂ω_ε₁),vec(∂ω_ε₂));
    ∂ω_εₑ_12 = @views reshape( j_εₑ_12 * v_∂ω, (3,3) );
    return εₑ_12, ∂ω_εₑ_12
end

function εₑ_∂ωεₑ_∂²ωεₑ(r₁,n,ε₁,ε₂,∂ω_ε₁,∂ω_ε₂,∂²ω_ε₁,∂²ω_ε₂)
    fjh_εₑ_12 = fjh_εₑ(vcat(r₁,n,vec(ε₁),vec(ε₂)));
    f_εₑ_12, j_εₑ_12, h_εₑ_12 = @views @inbounds fjh_εₑ_12[:,1], fjh_εₑ_12[:,2:23], reshape(fjh_εₑ_12[:,24:507],(9,22,22));
    εₑ_12 = @views reshape(f_εₑ_12,(3,3))
    v_∂ω, v_∂²ω = vcat(zeros(4),vec(∂ω_ε₁),vec(∂ω_ε₂)), vcat(zeros(4),vec(∂²ω_ε₁),vec(∂²ω_ε₂));
    ∂ω_εₑ_12 = @views reshape( j_εₑ_12 * v_∂ω, (3,3) );
    ∂ω²_εₑ_12 = @views reshape( [dot(v_∂ω,h_εₑ_12[i,:,:],v_∂ω) for i=1:9] + j_εₑ_12*v_∂²ω , (3,3) );
    return εₑ_12, ∂ω_εₑ_12, ∂ω²_εₑ_12
end






###############
jacCh_epse2DH = _jacCheck(f_epse2DH,x->fj_epse2DH(x)[:,2:15]);
jFD_2DH,jFM_2DH,jRM_2DH,jSym_2DH = jacCh_epse2DH(rand(14);nFD=3);
jacFD_2DH,jacFM_2DH,jacRM_2DH = _jacs(f_epse2DH)
p0_2DH = rand(14);
jres0_2DH = rand(9,15);
@benchmark $(jacFD_2DH)($p0_2DH)
@benchmark $(jacFM_2DH)($p0_2DH)
@benchmark $(jacRM_2DH)($p0_2DH)
@show @benchmark $(fj_epse2DH)($p0_2DH)
@show @benchmark $(fj_epse2DH!)($jres0_2DH,$p0_2DH)

function pb_f2DH!(res,x)
    res_copy = copy(res);
    ∂z_∂x = similar(x);
    f_epse2DH!(res,x);
    function f2DH!_pullback(∂z_∂res)
        Enzyme.autodiff(f_epse2DH!, Const, Duplicated(res_copy,∂z_∂res),Duplicated(x,∂z_∂x))
        return ∂z_∂x
    end
    return res, f2DH!_pullback
end

function pb_fj2DH!(res,x)
    res_copy = copy(res);
    ∂z_∂x = similar(x);
    fj_epse2DH!(res,x);
    function fj2DH!_pullback(∂z_∂res)
        Enzyme.autodiff(fj_epse2DH!, Const, Duplicated(res_copy,∂z_∂res),Duplicated(x,∂z_∂x))
        return ∂z_∂x
    end
    return res, fj2DH!_pullback
end

resf = rand(9);
resfj = rand(9,15);

outf,pbf = pb_f2DH!(resf,p0_2DH)
outfj,pbfj = pb_fj2DH!(resfj,p0_2DH)


ff1 = eval_fn_oop(f_epse2DH_sym,p2DH);
fj1 = eval_fn_oop(fj_epse2DH_sym,p2DH);

ff1(rand(14));
fj1(rand(14));

jacFDff1 = _jacFD(ff1);
jacFMff1 = _jacFM(ff1);
jacFDff1(rand(14));
jacFMff1(rand(14));

jacDFff1 = _jacDiffractor(ff1);
jacDF2ff1 = _jacDiffractor2(ff1);
jacDFff1(rand(14))
jacDF2ff1(rand(14));

jacChff1 = _jacCheck(ff1;nFD=5);
jFD1,jFM1,jRM1 = jacChff1(rand(14))


jacFDfj1 = _jacFD(fj1);
jacFMfj1 = _jacFM(fj1);
jacFDfj1(rand(14));
jacFMfj1(rand(14));
jacChfj1 = _jacCheck(fj1;nFD=5);
jFDfj1,jFMfj1,jRMfj1 = jacChfj1(rand(14))


f_epse2DH = eval_fn_oop(f_epse2DH_sym,p2DH);
fj_epse2DH = eval_fn_oop(fj_epse2DH_sym,p2DH);
# fjh_epse2DH = eval_fn_oop(fjh_epse2DH_sym,p2DH);
f_epse2DH(rand(14));
fj_epse2DH(rand(14));
# fjh_epse2DH(rand(14));


f_epse3DH_sym, p3DH = _f_epse3DH_sym();
fj_epse3DH_sym = _fj_sym(f_epse3DH_sym,p3DH);
# fj_epse3DH_sym, fjh_epse3DH_sym = _fj_fjh_sym(f_epse3DH_sym,p3DH);
fj_epse3DH = eval_fn_oop(fj_epse3DH_sym,p3DH);
# fjh_epse3DH = eval_fn_oop(fjh_epse3DH_sym,p3DH);
f_epse3DH = eval_fn_oop(f_epse3DH_sym,p3DH);
f_epse3DH(rand(16));
fj_epse3DH(rand(16));
# fjh_epse3DH(rand(16));

f_epse3D_sym, p3D = _f_epse3D_sym(); 
fj_epse3D_sym = _fj_sym(f_epse3D_sym,p3D);
fj_epse3D_sym, fjh_epse3D_sym = _fj_fjh_sym(f_epse3D_sym,p3D);
f_epse3D = eval_fn_oop(f_epse3D_sym,p3D);
fj_epse3D = eval_fn_oop(fj_epse3D_sym,p3D);
fjh_epse3D = eval_fn_oop(fjh_epse3D_sym,p3D);
f_epse3D(rand(22));
fj_epse3D(rand(22));
fjh_epse3D(rand(22));


# f_εₑ_sym, p = _f_epse3D_sym(); #_f_εₑ_sym();
# fj_εₑ_sym = _fj_sym(f_εₑ_sym,p);
# # fj_εₑ_sym, fjh_εₑ_sym = _fj_fjh_sym(f_εₑ_sym,p);
# f_εₑ = eval(fn_expr(f_εₑ_sym,p));
# fj_εₑ = eval(fn_expr(fj_εₑ_sym,p));
# # fjh_εₑ = eval(fn_expr(fjh_εₑ_sym,p));
# f_εₑ(rand(22));
# fj_εₑ(rand(22));
# # fjh_εₑ(rand(22));



### Utility functions to compute 
