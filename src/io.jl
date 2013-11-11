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
        ccall((@pysym :PyErr_SetString), Void, (PyPtr, Ptr{Uint8}),
              (pyexc::Dict)[PyIOError],
              bytestring(string("Julia exception: ", e)))
    else
        pyraise(e)
    end
end

function jl_IO_close(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        close(io)
        ccall((@pysym :Py_IncRef), Void, (PyPtr,), pynothing::PyPtr)
        return pynothing::PyPtr
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

# "closed" must be an attribute "for backwards compatibility"
function jl_IO_closed(self_::PyPtr, closure::Ptr{Void})
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        # Note: isopen is not defined for all IO subclasses in Julia.
        #       Should we do something different than throwing a MethodError,
        #       if isopen is not available?
        return pyincref(PyObject(!isopen(io))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_fileno(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        return pyincref(PyObject(fd(io))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_flush(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        if method_exists(flush, (typeof(io),))
            flush(io)
        end
        ccall((@pysym :Py_IncRef), Void, (PyPtr,), pynothing::PyPtr)
        return pynothing::PyPtr
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_isatty(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        return pyincref(PyObject(isa(io, Base.TTY))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_readable(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        return pyincref(PyObject(isreadable(io))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_writable(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        return pyincref(PyObject(iswritable(io))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_readline(self_::PyPtr, args_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO

        nargs = ccall((@pysym :PySequence_Size), Int, (PyPtr,), args_)
        nb = typemax(Int) # max #bytes to return
        if nargs == 1
            nb = convert(Int, PyObject(ccall((@pysym :PySequence_GetItem),
                                              PyPtr, (PyPtr, Int), args_, 0)))
            if nb < 0
                nb = typemax(Int)
            end
        elseif nargs > 1
            throw(ArgumentError("readline cannot accept $nargs arguments"))
        end

        d = readline(io)
        if length(d) > nb
            resize!(d, nb)
        end
        return pyincref(PyObject(d)).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_readlines(self_::PyPtr, args_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO

        nargs = ccall((@pysym :PySequence_Size), Int, (PyPtr,), args_)
        nb = typemax(Int) # max #bytes to return
        if nargs == 1
            nb = convert(Int, PyObject(ccall((@pysym :PySequence_GetItem),
                                              PyPtr, (PyPtr, Int), args_, 0)))
            if nb < 0
                nb = typemax(Int)
            end
        elseif nargs > 1
            throw(ArgumentError("readlines cannot accept $nargs arguments"))
        end

        ret = PyObject[]
        nread = 0
        while nread < nb && !eof(io)
            d = readline(io)
            nread += length(d)
            push!(ret, PyObject(d))
        end
        return pyincref(PyObject(ret)).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_seek(self_::PyPtr, args_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO

        nargs = ccall((@pysym :PySequence_Size), Int, (PyPtr,), args_)
        if nargs < 1 || nargs > 2
            throw(ArgumentError("seek cannot accept $nargs arguments"))
        end
        offset = convert(FileOffset, 
                         PyObject(ccall((@pysym :PySequence_GetItem),
                                        PyPtr, (PyPtr, Int), args_, 0)))
        whence = nargs == 1 ? 0 :
          convert(Int, PyObject(ccall((@pysym :PySequence_GetItem),
                                      PyPtr, (PyPtr, Int), args_, 1)))


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
        return pyincref(PyObject(position(io))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

isseekable(io) = method_exists(seek, (typeof(io), FileOffset))
isseekable(io::IOBuffer) = io.seekable

function jl_IO_seekable(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        return pyincref(PyObject(isseekable(io))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_tell(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        return pyincref(PyObject(position(io))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_writelines(self_::PyPtr, arg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        for s in PyVector{String}(pyincref(arg_))
            write(io, s)
        end
        ccall((@pysym :Py_IncRef), Void, (PyPtr,), pynothing::PyPtr)
        return pynothing::PyPtr
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

##########################################################################
# RawIOBase methods:

function jl_IO_read(self_::PyPtr, args_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO

        nargs = ccall((@pysym :PySequence_Size), Int, (PyPtr,), args_)
        if nargs > 1
            throw(ArgumentError("read cannot accept $nargs arguments"))
        end
        nb = nargs == 0 ? -1 :
          convert(Int, PyObject(ccall((@pysym :PySequence_GetItem),
                                      PyPtr, (PyPtr, Int), args_, 0)))
        if nb < 0
            nb = typemax(Int)
        end

        return pyincref(PyObject(readbytes(io, nb))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_readall(self_::PyPtr, noarg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        return pyincref(PyObject(readbytes(io))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_readinto(self_::PyPtr, arg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        b = PyVector{Uint8}(pyincref(arg_))
        return pyincref(PyObject(readbytes!(io, b))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

function jl_IO_write(self_::PyPtr, arg_::PyPtr)
    try
        io = unsafe_pyjlwrap_to_objref(self_)::IO
        b = convert(Vector{Uint8}, pyincref(arg_))
        return pyincref(PyObject(write(io, b))).o
    catch e
        ioraise(e)
    end
    return convert(PyPtr, C_NULL)
end

##########################################################################
# TODO: support other Python interfaces (e.g. TextIO) when possible?

##########################################################################

const jl_IO_methods = PyMethodDef[
PyMethodDef("close", jl_IO_close, METH_NOARGS),
PyMethodDef("fileno", jl_IO_fileno, METH_NOARGS),
PyMethodDef("flush", jl_IO_flush, METH_NOARGS),
PyMethodDef("isatty", jl_IO_isatty, METH_NOARGS),
PyMethodDef("readable", jl_IO_readable, METH_NOARGS),
PyMethodDef("writable", jl_IO_writable, METH_NOARGS),
PyMethodDef("readline", jl_IO_readline, METH_VARARGS),
PyMethodDef("readlines", jl_IO_readlines, METH_VARARGS),
PyMethodDef("seek", jl_IO_seek, METH_VARARGS),
PyMethodDef("seekable", jl_IO_seekable, METH_NOARGS),
PyMethodDef("tell", jl_IO_tell, METH_NOARGS),
PyMethodDef("writelines", jl_IO_writelines, METH_O),
PyMethodDef("read", jl_IO_read, METH_VARARGS),
PyMethodDef("readall", jl_IO_readall, METH_NOARGS),
PyMethodDef("readinto", jl_IO_readinto, METH_O),
PyMethodDef("write", jl_IO_write, METH_O),
PyMethodDef() # sentinel
]

const jl_IO_getset = PyGetSetDef[
PyGetSetDef("closed", jl_IO_closed)
PyGetSetDef()
]

function pyio_repr(o::PyPtr)
    o = PyObject(try string("<PyCall.io ",unsafe_pyjlwrap_to_objref(o),">")
                 catch "<PyCall.io NULL>"; end)
    oret = o.o
    o.o = convert(PyPtr, C_NULL) # don't decref
    return oret
end
const pyio_repr_ptr = cfunction(pyio_repr, PyPtr, (PyPtr,))

jl_IOType = PyTypeObject()
function pyio_initialize()
    global jl_IOType
    if (jl_IOType::PyTypeObject).tp_name == C_NULL
        jl_IOType::PyTypeObject =
        pyjlwrap_type("PyCall.jl_IO",
                      t -> begin
                          t.tp_getattro = pysym(:PyObject_GenericGetAttr)
                          t.tp_methods = pointer(jl_IO_methods)
                          t.tp_getset = pointer(jl_IO_getset)
                          t.tp_repr = pyio_repr_ptr
                      end)
    end
    return
end

function pyio_finalize()
    global jl_IOType
    jl_IOType::PyTypeObject = PyTypeObject()
end

##########################################################################

function PyObject(io::IO)
    global jl_IOType
    if (jl_IOType::PyTypeObject).tp_name == C_NULL
        pyio_initialize()
    end
    pyjlwrap_new(jl_IOType::PyTypeObject, io)
end
