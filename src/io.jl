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

function create_data_file(fname,datasets,sizes,types; data_dir=joinpath(homedir(),"data"), groups=nothing)
	fpath 	= joinpath(data_dir, fname)
	fid 	= h5open(fpath, "cw")			# "cw"== "read-write, create if not existing"
	if !isnothing(groups)
		group_refs = [create_group(fid, gr_name) for gr_name in groups]
        ds_refs = [create_dataset(fid, gr*"/"*ds, randn(tt,sz...)) for gr in groups, (ds,sz,tt) in zip(datasets,sizes,types)]
    else
        ds_refs = [create_dataset(fid, ds, randn(tt,sz...)) for (ds,sz,tt) in zip(datasets,sizes,types)]
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
