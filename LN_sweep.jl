## parameter sweep for comparison with optimization trajectories
using OptiMode
using HDF5
using Zygote: @ignore
using Base.Filesystem

# task_id = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

function sum_Δng²_FHSH(ωs, p; Δx=6.0,Δy=4.0,Nx=128,Ny=128, fpath=nothing, grp_name=nothing,
	pidx=nothing,verbose=true)
	grid = Grid(Δx,Δy,Nx,Ny)
	rwg_pe(x) = ridge_wg_partial_etch(x[1],x[2],x[3],x[4],0.5,MgO_LiNbO₃,SiO₂,Δx,Δy) # partially etched ridge waveguide with dispersive materials, x[3] is partial etch fraction of top layer, x[3]*x[2] is etch depth, remaining top layer thickness = x[2]*(1-x[3]).
	nω = length(ωs)
    ngs_FHSH = solve_n(
				vcat(ωs, 2*ωs ),
				rwg_pe(p),
				grid,
			)[2]
    ngs_FH = ngs_FHSH[1:nω]
	ngs_SH = ngs_FHSH[nω+1:2*nω]
	Δng² = abs2.(ngs_SH .- ngs_FH)
	@ignore begin
		# print results for job log
		if verbose
			println("")
			println("pidx: $pidx")
			println("p: $p")
			println("\tngs_FH: $ngs_FH")
			println("\tngs_SH: $ngs_SH")
			println("\tsum(Δng²): $(sum(Δng²))")
			println("")
		end
		# save results to a data file (if provided)
		if !isnothing(fpath)
			grp_name = isnothing(grp_name) ? "/" : grp_name*"/"
			h5open(fpath,"r+") do fid
				fid[grp_name*"ngs_FH"][:,pidx.I...]		=	ngs_FH
				fid[grp_name*"ngs_SH"][:,pidx.I...]		=	ngs_SH
				fid[grp_name*"sumΔng²"][pidx.I...]	=	sum(Δng²)
			end
		end
	end
    return sum(Δng²)
end

"""
	create_data_file(fname; data_dir, groups, datasets)

Create a data file with filename `fname` in directory `data_dir` and return the path.

Currently this always creates HDF5 files, for which optional inputs `groups` and `datasets`
specify names of group and dataset objects to create in the new HDF5 file.
"""
function create_data_file(fname; data_dir=joinpath(homedir(),"data"), groups=nothing,
	datasets=nothing)
	fpath 	= joinpath(data_dir, fname)
	fid 	= h5open(fpath, "cw")			# "cw"== "read-write, create if not existing"
	if !isnothing(groups)
		group_refs = [create_group(fid, gr_name) for gr_name in groups]
		ds_refs = [create_dataset(fid, ds_name) for ds_name in datasets]
	end
	close(fid)
	return fpath
end


function sweep_params(p_upper,p_lower,nps,ωs; fname=nothing, data_dir=joinpath(homedir(),"data"), grp_name=nothing)
	ranges = [range(p_lower[i],p_upper[i],length=nps[i]) for i=1:length(nps)]
	ps = collect(Iterators.product(ranges...)) # collect(Iterators.product(reverse(ranges)...))

	nω = length(ωs)
	fpath 	= joinpath(data_dir, fname)
	h5open(fpath, "cw") do fid			# "cw"== "read-write, create if not existing"
		ds_names = ["ngs_FH", "ngs_SH", "sumΔng²"]
		ds_sizes = [(nω,size(ps)...),(nω,size(ps)...),size(ps) ]
		ds_refs = [create_dataset(fid, nm, rand(sz...)) for (nm,sz) in zip(ds_names,ds_sizes)]
	end

	for pidx in CartesianIndices(ps)
		pp = ps[pidx]
		try
			sum_Δng²_FHSH(ωs,pp; pidx, fpath, grp_name)
		catch
			println("")
			println("pidx: $pidx")
			println("p: $pp")
			println("\tngs_FH: Error")
			println("\tngs_SH: Error")
			println("\tsum(Δng²): Error")
			println("")
		end
	end
end

λs = reverse(1.4:0.01:1.6)
ωs = 1 ./ λs
p_lower 	= 	[	0.4,	0.3, 	0., 	0.,		]
p_upper 	= 	[	2., 	2., 	1., 	π/4.,	]
nps			=	[	20, 	20, 	5, 		3,		]
# nps			=	[	2, 	2, 	2, 		2,		]
sweep_params(p_upper,p_lower,nps,ωs;fname="sweep_LN_test3.h5")
