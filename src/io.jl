using Pkg
using LibGit2
using Dates: now
export modified_untracked_deleted_files, push_local_src, timestamp_string

timestamp_string() = string(now())

"""
Return the paths of all modified or untracked files git finds in the subdirectory `subdir` of 
git repository `repo`
"""
function modified_untracked_deleted_files(repo::LibGit2.GitRepo,subdir="src";verbose=false)
	repo_path = LibGit2.path(repo) # LibGit2.workdir(repo)
	tree = LibGit2.GitTree(repo,"HEAD^{tree}")
 	diff = LibGit2.diff_tree(repo,tree,"")
	if verbose
		println("#### checking modified & deleted files in $subdir using diff #####")
	end
	changed_files_diff = []
	for i in 1:LibGit2.count(diff)
		d = diff[i]   # print(d)
		old_file_path = unsafe_string(d.old_file.path) # if unsafe_string causes problems consider something like:
		new_file_path = unsafe_string(d.new_file.path) # GC.@preserve text s = unsafe_string(pointer(text))
		if verbose
			println("diff[$i]: \n", "old file path: ", old_file_path, "\n new file path: ",new_file_path,"\n---\n")
		end
		if isnothing(subdir) || isequal(first(splitpath(new_file_path)),subdir)
			push!(changed_files_diff,new_file_path)
		end
	end
	
	if verbose
		println("#### checking modified & untracked files in $subdir using walkdir and GitLib2.status #####")
	end
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
	modified_files = intersect(changed_files_diff,changed_files)
	untracked_files = setdiff(changed_files,changed_files_diff)
	deleted_files = setdiff(changed_files_diff,changed_files)
	return modified_files, untracked_files, deleted_files
end
modified_untracked_deleted_files(repo_path::AbstractString,subdir="src") = modified_untracked_deleted_files(LibGit2.GitRepo(repo_path),subdir)
modified_untracked_deleted_files(imported_module::Module,subdir="src") = modified_untracked_deleted_files(dirname(dirname(pathof(imported_module))),subdir)

function push_local_src(;package=@__MODULE__,verbose=false)
	# package_path = Base.find_package(string(package)) # Doesn't require package to be imported
	package_path = pathof(package)	# requires package to be imported if not called from within
	src_path = dirname(package_path)
	repo_path = dirname(src_path)
	repo = LibGit2.GitRepo(repo_path)	# create GitRepo object for package 
	modified_files, untracked_files, deleted_files = modified_untracked_deleted_files(repo,"src";verbose)
	# manually look up any other required files outside of `src` directory to `modified_files` if they have changed (nonzero `LibGit2.status`)
	for file_relpath in ("Project.toml","Manifest.toml")
		if !iszero(LibGit2.status(repo,file_relpath))
			push!(modified_files,file_relpath)
		end
	end
	add_files = vcat(modified_files,untracked_files,deleted_files) # stage all additions, modifications and deletions of files in "src" directory
	if length(add_files)>0
		LibGit2.add!(repo,add_files...)	
		msg = "auto-pushing local source code changes. " * timestamp_string()
		if verbose
			println("\n#########  auto-pushing local source code changes  ##########\n")
			println("\tcommit message: ",msg)
			println("\tchanged files to commit and push: ")
			println("\t\tmodified files: ")
			[println("\t\t\t"*ff) for ff in modified_files]
			println("\t\tuntracked files: ")
			[println("\t\t\t"*ff) for ff in untracked_files]
			println("\t\tdeleted files: ")
			[println("\t\t\t"*ff) for ff in deleted_files]
		end
		LibGit2.commit(repo,msg)	# commit all staged changes
		# remote = LibGit2.get(LibGit2.GitRemote, repo, "origin")
		LibGit2.push( # push auto-generated commmit containing local changes to files in src to remote
			repo,
			refspecs=["refs/heads/main"],
			remote="origin",
			force=true,
		)	
	else
		if verbose 
			println("\n#########  no local source code changes to commit/push detected  ##########\n")
		end
	end
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
