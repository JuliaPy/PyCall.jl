# This is an alternative definition of PyObject(::IO), using @pydef.
# At the end of this file, we call runtests.jl to check that the tests still
# pass with this definition.



# If we wanted to support TextIO with @pydef, there's a few solutions:
# A. Create a different Python type for it. That's not really possible with
#    @pydef at the moment
# B. Support PyMemberDef, add _istextio as a Python class member
# C. Use a WeakKeyDict to store which IO object is a TextIO. It's awkward, but
#    probably good enough.
# This is only a problem because we can't add a field to Base.IO, it's unlikely
# that many PyCall users will have that problem.
istextio(io::IO) = false # for now

PyCall.@pydef type IO
    close(io) = close(io)
    closed.get(io) = !isopen(io)
    encoding.get(io) = "UTF-8"
    fileno(io) = fd(io)
    flush(io) = flush(io)
    isatty(io) = isa(io, Base.TTY)
    readable(io) = isreadable(io)
    writable(io) = iswritable(io)
    readline(io, size=typemax(Int)) = jl_readline(io, size)
    readlines(io, size=typemax(Int)) = jl_readlines(io, size)
    seek(io, offset, whence=1) = jl_seek(io, offset, whence)
    seekable(io) = PyCall.isseekable(io)
    tell(io) = position(io)
    writelines(io, seq) = for s in seq write(io, s) end
    read(io, nb=typemax) =
        istextio(io) ? bytestring(readbytes(io, nb)) : readbytes(io, nb)
    readall(io) = istextio(io) ? readall(io) : readbytes(io)
    readinto(io, b) = readbytes!(io, b)
    write(io, b) = write(io, b)
end


function jl_readline(io::IO, nb)
    d = readline(io)
    if length(d) > nb
        resize!(d, nb)
    end
    return d
end

function jl_readlines(io::IO, nb)
    ret = Any[]  # should it be some other type?
    nread = 0
    while nread < nb && !eof(io)
        d = readline(io)
        nread += length(d)
        push!(ret, PyObject(d))
    end
    return ret
end

function jl_seek(io::IO, offset, whence)
    if whence == 0
        seek(io, offset)
    elseif whence == 1
        skip(io, offset)
    elseif whence == 2
        seekend(io)
        skip(io, offset)
    else
        throw(ArgumentError("unrecognized whence=$n argument to seek"))
    end
    return position(io)
end



include("runtests.jl")
