# logger based on Tamas Papp's HDF5Logging.jl:
# https://github.com/tpapp/HDF5Logging.jl/blob/master/src/HDF5Logging.jl

# module HDF5Logging

export hdf5_logger

using DocStringExtensions: SIGNATURES, TYPEDEF
import Logging
using HDF5: h5open, create_group

"""
$(TYPEDEF)
Logger into HDF5 files. Use [`hdf5_logger`](@ref) to create.
"""
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

"""
$(SIGNATURES)
Utility function to find the last index, ie the highest group name that parses as an
integer. Internal.
"""
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

"""
$(SIGNATURES)
Create a logger that write log messages into the HDF5 file `filename` within the given
`group_path` (defaults to `"log"`).
# Logging
A counter keeps track of an increasing integer index. A log message is written to the given
`group_path` as a group named with this index (converted to a string), with the following
fields:
- `level::Int`, `message::String`, `_module::String`, `group::String`, `id::String`,
  `file::String`, `line::String`, which are part of every log message
- `data`, which contains additional key-value pairs as passed on by the user. Keys are
  strings.
# Reading logs
`length(logger)` returns the last index of logged messages, which can be be accessed with
`logger[i]`. The latter returns a `NamedTuple`, or `nothing` for no such message.
# Example
```jldoctest; filter = [r"group = .*", r"file \\S*"]
julia> using HDF5Logging, Logging
julia> logger = hdf5_logger(tempname())
Logging into HDF5 file /tmp/jl_IbbUvj, 0 messages in “log”
julia> # write log
julia> with_logger(logger) do
       @info "very informative" a = 1
       end
julia> # read log
julia> logger[1]
(level = Info, message = "very informative", _module = "Main", group = "REPL[46]", id = "Main_7a40b9cc", file = "REPL[46]", line = 2, data = ["a" => 1])
```
# Notes
1. The HDF5 file can contain other data, ideally in other groups than `group_path`.
2. Contiguity of message indexes is not checked. This package will create them in order,
starting at 1, but if you delete some with another tool then `getindex` will just return
`nothing`.
3. A lock is used, so a shared instance should be thread-safe. That said, **if you open the
same file with another `hdf5_logger`, consequences are undefined.**
4. The HDF5 file is not kept open when not accessed. This is slower, but should help ensure
robust operation.
"""
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

Logging.min_enabled_level(::HDF5Logger) = 0

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