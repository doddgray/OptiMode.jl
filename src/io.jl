using Pkg
using LibGit2
using Dates: now
export modified_files, push_local_src, timestamp_string

timestamp_string() = string(now())

"""
Return the paths of all modified or untracked files git finds in the subdirectory `subdir` of 
git repository `repo`
"""
function modified_files(repo::LibGit2.GitRepo,subdir="src")
	repo_path = LibGit2.path(repo) # LibGit2.workdir(repo)
	tree = LibGit2.GitTree(repo,"HEAD^{tree}")
 	diff = LibGit2.diff_tree(repo,tree,"")
	
	println("#### checking modified & deleted files using diff #####")
	changed_files_diff = []
	for i in 1:LibGit2.count(diff)
		d = diff[i]   # print(d)
		old_file_path = unsafe_string(d.old_file.path) # if unsafe_string causes problems consider something like:
		new_file_path = unsafe_string(d.new_file.path) # GC.@preserve text s = unsafe_string(pointer(text))
		println("diff[$i]: \n", "old file path: ", old_file_path, "\n new file path: ",new_file_path,"\n---\n")
		if isnothing(subdir) || isequal(first(splitpath(new_file_path)),subdir)
			push!(changed_files_diff,new_file_path)
		end
	end
	
	println("#### checking modified & untracked files using walkdir and GitLib2.status #####")
	changed_files = []
	for (root, dirs, files) in walkdir(joinpath(repo_path,subdir))
		for file in files
			file_path = joinpath(root, file) 
			rel_path = relpath(file_path, repo_path)
			file_status = LibGit2.status(repo,rel_path) # Nonzero if file is modified or untracked
			if !iszero(file_status)
				push!(changed_files, rel_path)
			end 
		end
	end
	return changed_files_diff, changed_files
end
modified_files(repo_path::AbstractString,subdir="src") = modified_files(LibGit2.GitRepo(repo_path),subdir)
modified_files(imported_module::Module,subdir="src") = modified_files(dirname(dirname(pathof(imported_module))),subdir)

function push_local_src(;package=@__MODULE__)
	# package_path = Base.find_package(string(package)) # Doesn't require package to be imported
	package_path = pathof(package)	# requires package to be imported if not called from within
	src_path = dirname(package_path)
	repo_path = dirname(src_path)
	repo = LibGit2.GitRepo(repo_path)	# create GitRepo object for package 
	changed_files_diff, changed_files = modified_files(repo,"src")
	LibGit2.add!(repo,changed_files...)	# Add all new or modified files in "src" directory to upcoming git commit
	msg = "auto-pushed local source code changes. " * timestamp_string()
	println(msg)
	println("changed files: ")
	[println("\t"*ff) for ff in changed_files]
	LibGit2.commit(repo,msg)	# Add all new or modified files in "src" directory to upcoming git commit
	remote = LibGit2.get(LibGit2.GitRemote, repo, "origin")
	LibGit2.push( # push auto-generated commmit containing local changes to files in src to remote
		repo,
		# refspecs=["refs/remotes/origin/main"], # 
		refspecs=["refs/heads/main"],
		remote="origin/main",
		force=true,
	)	
	# test modification
end

# function sync_repo(remote_url)

# end






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
