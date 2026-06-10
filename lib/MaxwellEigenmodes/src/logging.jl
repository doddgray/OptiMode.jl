# logger based on Tamas Papp's HDF5Logging.jl:
# https://github.com/tpapp/HDF5Logging.jl/blob/master/src/HDF5Logging.jl

# module HDF5Logging

export hdf5_logger, HDF5Logger

import Logging
using HDF5: h5open, create_group



# """
# $(TYPEDEF)
# Logger into HDF5 files. Use [`hdf5_logger`](@ref) to create.
# """
mutable struct HDF5Logger{L} <: Logging.AbstractLogger
    "The filename of a HDF5 file."
    filename::String
    "The group path within the HDF5 file for logs."
    group_path::String
    "Index of last log item."
    last_index::Int
    "Lock for writes."
    lock::L
end

function Base.show(io::IO, logger::HDF5Logger)
    print(io, "Logging into HDF5 file ", logger.filename, ", ", logger.last_index,
          " messages in “", logger.group_path, "”")
end

# """
# $(SIGNATURES)
# Utility function to find the last index, ie the highest group name that parses as an
# integer. Internal.
# """
function get_last_index(log_group)
    last_index = 0
    for k in keys(log_group)
        i = tryparse(Int, k)
        if i ≠ nothing
            last_index = max(last_index, i)
        end
    end
    last_index
end

function hdf5_logger(filename; group_path = "log")
    last_index = 0
    h5open(filename, "cw") do fid
        if haskey(fid, group_path)
            last_index = get_last_index(fid[group_path])
        else
            create_group(fid, group_path)
        end
    end
    HDF5Logger(filename, group_path, last_index, ReentrantLock())
end

####
#### write logs
####

function Logging.handle_message(logger::HDF5Logger, level, message, _module, group, id,
                                file, line; kwargs...)
    lock(logger.lock)
    try
        h5open(logger.filename, "r+") do fid
            log_group = fid[logger.group_path]
            logger.last_index += 1
            log_key = string(logger.last_index)
            @assert !haskey(log_group, log_key)
            msg = create_group(log_group, log_key)
            msg["level"] = Int(level.level)
            msg["message"] = string(message)
            msg["_module"] = string(_module)
            msg["group"] = string(group)
            msg["id"] = string(id)
            msg["file"] = string(file)
            msg["line"] = Int(line)
            data = create_group(msg, "data")
            for (k, v) in kwargs
                data[string(k)] = v
            end
        end
    finally
        unlock(logger.lock)
    end
end

Logging.shouldlog(::HDF5Logger, level, _module, group, id) = true

Logging.min_enabled_level(::HDF5Logger) = Logging.Info

####
#### read logs
####

Base.length(logger::HDF5Logger) = logger.last_index

function Base.getindex(logger::HDF5Logger, i::Integer)
    h5open(logger.filename, "r+") do fid
        log_group = fid[logger.group_path]
        log_key = string(i)
        if haskey(log_group, log_key)
            msg = log_group[log_key]
            (level = Logging.LogLevel(msg["level"][]),
             message = msg["message"][],
             _module = msg["_module"][],
             group = msg["group"][],
             id = msg["id"][],
             file = msg["file"][],
             line = msg["line"][],
             data = [k => v[] for (k, v) in pairs(msg["data"])])
        else
            nothing
        end
    end
end

# end # module