# Julia IO subclasses are converted into Python objects implementing
# the IOBase + RawIOBase interface.
#
# (A useful model here is the Python FileIO class, which is implemented
#  in Modules/_io/fileio.c in the Python source code.)

##########################################################################
# IOBase methods:

# IO objects should raise IOError for unsupported operations or failed IO
function ioraise(e)
    if isa(e, MethodError) || isa(e, ErrorException)
        ccall((@pysym :PyErr_SetString), Cvoid, (PyPtr, Cstring),
              (pyexc::Dict)[PyIOError],
              string("Julia exception: ", e))
    else
        pyraise(e)
    end
end

macro with_ioraise(expr)
    :(try
        $(esc(expr))
      catch e
        ioraise(e)
      end)
end

function jl_io_readline(io::IO, nb)
    d = Compat.readline(io, chomp=false)
    if sizeof(d) > nb ≥ 0
        error("byte-limited readline is not yet supported by PyCall")
    end
    return d
end

function jl_io_readlines(io::IO, nb)
    ret = PyObject[]
    nread = 0
    while (nb < 0 || nread ≤ nb) && !eof(io)
        d = Compat.readline(io, chomp=false)
        nread += sizeof(d)
        push!(ret, PyObject(d))
    end
    return ret
end

isseekable(io) = method_exists(seek, Tuple{typeof(io), Int64})
isseekable(io::IOBuffer) = io.seekable

function jl_io_seek(io::IO, offset, whence)
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


##########################################################################

pyio_jl(self::PyObject) = unsafe_pyjlwrap_to_objref(self["io"].o)::IO

const PyIO = PyNULL()

pyio_initialized = false
function pyio_initialize()
    global pyio_initialized
    if !pyio_initialized::Bool
        copy!(PyIO, @pydef_object mutable struct PyIO
            __init__(self, io::IO; istextio=false) = begin
                self[:io] = pyjlwrap_new(io) # avoid recursion
                self[:istextio] = istextio
            end
            close(self) = @with_ioraise(close(pyio_jl(self)))
            closed.get(self) = @with_ioraise(!isopen(pyio_jl(self)))
            encoding.get(self) = "UTF-8"
            fileno(self) = @with_ioraise(fd(pyio_jl(self)))
            flush(self) = begin
                @with_ioraise begin
                    io = pyio_jl(self)
                    if method_exists(flush, Tuple{typeof(io)})
                        flush(io)
                    end
                end
            end
            isatty(self) = isa(pyio_jl(self), Base.TTY)
            readable(self) = isreadable(pyio_jl(self))
            writable(self) = iswritable(pyio_jl(self))
            readline(self, size=typemax(Int)) =
                @with_ioraise(jl_io_readline(pyio_jl(self), size))
            readlines(self, size=typemax(Int)) =
                @with_ioraise(array2py(jl_io_readlines(pyio_jl(self), size)))
            seek(self, offset, whence=1) =
                @with_ioraise(jl_io_seek(pyio_jl(self), offset, whence))
            seekable(self) = isseekable(pyio_jl(self))
            tell(self) = @with_ioraise(position(pyio_jl(self)))
            writelines(self, seq) =
                @with_ioraise(for s in seq write(pyio_jl(self), s) end)
            read(self, nb=typemax(Int)) =
                @with_ioraise(self[:istextio] ?
                              String(read(pyio_jl(self), nb < 0 ? typemax(Int) : nb)) :
                              pybytes(read(pyio_jl(self), nb < 0 ? typemax(Int) : nb)))
            readall(self) =
                @with_ioraise(self[:istextio] ? read(pyio_jl(self), String) :
                                                pybytes(read(pyio_jl(self))))
            readinto(self, b) = @with_ioraise(pybytes(readbytes!(pyio_jl(self), b)))
            write(self, b) = @with_ioraise(write(pyio_jl(self), b))
        end)
        pyio_initialized = true
    end
    return
end

##########################################################################

function PyObject(io::IO)
    pyio_initialize()
    # pyjlwrap_new is necessary to avoid PyIO(io) calling PyObject(::IO)
    PyIO(pyjlwrap_new(io))
end

"""
    PyTextIO(io::IO)
    PyObject(io::IO)

Julia IO streams are converted into Python objects implementing the RawIOBase interface,
so they can be used for binary I/O in Python
"""
function PyTextIO(io::IO)
    pyio_initialize()
    PyIO(pyjlwrap_new(io); istextio=true)
end
