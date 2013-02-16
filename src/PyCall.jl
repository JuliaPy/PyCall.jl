module PyCall

export pyinitialize, pyfinalize, pycall, pyimport, pyglobal, PyObject, pyfunc, PyPtr

import Base.convert
import Base.ref
import Base.show

typealias PyPtr Ptr{Void} # type for PythonObject* in ccall

#########################################################################

# Global configuration variables.  Note that, since Julia does not allow us
# to attach type annotations to globals, we need to annotate these explicitly
# as initialized::Bool and libpython::String when we use them.
initialized = false # whether Python is initialized
libpython = C_NULL # Python shared library (from dlopen)

libpython_name(python::String) =
  replace(readall(`$python -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LDLIBRARY')"`),
          r"\.so(\.[0-9\.]+)?\s*$|\.dll\s*$", "", 1)

pyfunc(func::Symbol) = dlsym(libpython::Ptr{Void}, func)

# initialize the Python interpreter (no-op on subsequent calls)
function pyinitialize(python::String)
    global initialized
    global libpython
    if (!initialized::Bool)
        libpython::Ptr{Void} = dlopen_global(libpython_name(python))
        ccall(pyfunc(:Py_InitializeEx), Void, (Int32,), 0)
        initialized::Bool = true
    end
    nothing
end

pyinitialize() = pyinitialize("python") # default Python executable name
libpython_name() = libpython_name("python") # ditto

# end the Python interpreter and free associated memory
function pyfinalize()
    global initialized
    global libpython
    if (initialized::Bool)
        ccall(pyfunc(:Py_Finalize), Void, ())
        dlclose(libpython::Ptr{Void})
        libpython::Ptr{Void} = C_NULL
        initialized::Bool = false
    end
    nothing
end

#########################################################################
# Wrapper around Python's C PyObject* type, with hooks to Python reference
# counting and conversion routines to/from C and Julia types.

type PyObject
    o::PyPtr
    function PyObject(o::PyPtr)
        po = new(o)
        finalizer(po, po -> ccall(pyfunc(:Py_DecRef), 
                                  Void, (PyPtr,), po.o))
        return po
    end
end

# conversion to pass PyObject as ccall arguments:
convert(::Type{PyPtr}, po::PyObject) = po.o

# use constructor for generic conversions to PyObject
convert(::Type{PyObject}, o) = PyObject(o)
PyObject(o::PyObject) = o

#########################################################################
# Conversions of simple types (numbers and strings)

# conversions from Julia types to PyObject:

PyObject(i::Unsigned) = PyObject(ccall(pyfunc(:PyInt_FromSize_t),
                                       PyPtr, (Uint,), i))
PyObject(i::Integer) = PyObject(ccall(pyfunc(:PyInt_FromSsize_t),
                                      PyPtr, (Int,), i))

PyObject(b::Bool) = 
  PyObject(ccall(pyfunc(:PyBool_FromLong),
                 PyPtr, (OS_NAME == :Windows ? Int32 : Int,), b))

PyObject(r::Real) = PyObject(ccall(pyfunc(:PyFloat_FromDouble),
                                   PyPtr, (Float64,), r))

PyObject(c::Complex) = PyObject(ccall(pyfunc(:PyComplex_FromDoubles),
                                      PyPtr, (Float64,Float64), 
                                      real(c), imag(c)))

# fixme: PyString_* was renamed to PyBytes_* in Python 3.x?
PyObject(s::String) = PyObject(ccall(pyfunc(:PyString_FromString),
                                     PyPtr, (Ptr{Uint8},), bytestring(s)))

# conversions to Julia types from PyObject

convert{T<:Integer}(::Type{T}, po::PyObject) = 
  convert(T, ccall(pyfunc(:PyInt_AsSsize_t), Int, (PyPtr,), po))

convert(::Type{Bool}, po::PyObject) = 
  convert(Bool, ccall(pyfunc(:PyInt_AsSsize_t), Int, (PyPtr,), po))

convert{T<:Real}(::Type{T}, po::PyObject) = 
  convert(T, ccall(pyfunc(:PyFloat_AsDouble), Float64, (PyPtr,), po))

convert{T<:Complex}(::Type{T}, po::PyObject) = 
  convert(T, 
          complex128(ccall(pyfunc(:PyComplex_RealAsDouble), 
                           Float64, (PyPtr,), po),
                     ccall(pyfunc(:PyComplex_ImagAsDouble), 
                           Float64, (PyPtr,), po)))

convert(::Type{String}, po::PyObject) =
  bytestring(ccall(pyfunc(:PyString_AsString), 
                   Ptr{Uint8}, (PyPtr,), po))

#########################################################################
# Tuple conversion

function PyObject(t::(Any...)) 
    o = ccall(pyfunc(:PyTuple_New), PyPtr, (Int,), length(t))
    if o == C_NULL
        error("failure creating Python tuple")
    end
    for i = 1:length(t)
        oi = PyObject(t[i])
        if 0 != ccall(pyfunc(:PyTuple_SetItem), Int32, (PyPtr,Int,PyPtr),
                      o, i-1, oi)
            error("error setting Python tuple")
        end
        oi.o = C_NULL # PyTuple_SetItem steals the reference
    end
    return PyObject(o)
end

convert(tt::(Type...), o::PyObject) = 
  ntuple(ccall(pyfunc(:PyTuple_Size), Int, (PyPtr,), o), i -> begin
      oi = PyObject(ccall(pyfunc(:PyTuple_GetItem), PyPtr, (PyPtr,Int), o,i-1))
      ti = convert(tt[i], oi)
      oi.o = C_NULL # GetItem returns borrowed reference
      ti
  end)

#########################################################################
# Pretty-printing PyObject

function show(io::IO, o::PyObject)
    if o.o == C_NULL
        print(io, "PyObject NULL")
    else
        s = ccall(pyfunc(:PyObject_Str), PyPtr, (PyPtr,), o)
        if (s == C_NULL)
            s = ccall(pyfunc(:PyObject_Repr), PyPtr, (PyPtr,), o)
            if (s == C_NULL)
                return print(io, "PyObject $o.o")
            end
        end
        print(io, "PyObject $(convert(String, PyObject(s)))")
    end
end

#########################################################################
# For o::PyObject, make o["foo"] and o[:foo] equivalent to o.foo in Python

function ref(o::PyObject, s::String)
    p = ccall(pyfunc(:PyObject_GetAttrString), PyPtr,
              (PyPtr, Ptr{Uint8}), o, bytestring(s))
    if p == C_NULL
        throw(KeyError(s))
    end
    return PyObject(p)
end

ref(o::PyObject, s::Symbol) = ref(o, string(s))

#########################################################################

function pyimport(name::String)
    pyinitialize()
    mod = PyObject(ccall(pyfunc(:PyImport_ImportModule), PyPtr,
                         (Ptr{Uint8},), bytestring(name)))
    # fixme: check for errors
    return mod
end

pyimport(name::Symbol) = pyimport(string(name))

# look up a global variable (in module __main__)
function pyglobal(name::String)
    pyinitialize()
    # fixme: check for errors
    return PyObject(ccall(pyfunc(:PyObject_GetAttrString), PyPtr,
                          (PyPtr, Ptr{Uint8}), 
                          ccall(pyfunc(:PyImport_AddModule), PyPtr,
                                (Ptr{Uint8},), bytestring("__main__")),
                          bytestring(name)))
end

pyglobal(name::Symbol) = pyglobal(string(name))

#########################################################################

function pycall(o::PyObject, returntype::Union(Type,(Type...)), args...)
    pyinitialize()
    oargs = map(PyObject, args)
    # would rather call PyTuple_Pack, but calling varargs functions
    # with ccall and argument splicing seems problematic right now.
    arg = PyObject(ccall(pyfunc(:PyTuple_New), PyPtr, (Int,), 
                         length(args)))
    if arg.o == C_NULL
        error("failure creating Python argument tuple")
    end
    for i = 1:length(args)
        if 0 != ccall(pyfunc(:PyTuple_SetItem), Int32, (PyPtr,Int,PyPtr),
                      arg, i-1, oargs[i])
            error("error setting Python argument tuple")
        end
        # PyTuple_SetItem steals the reference.  (Note: Don't
        # set oargs[i].o to C_NULL here, since the original arg
        # might itself have been a PyObject and we don't want
        # it to "lose" its object.  IncRef instead.)
        ccall(pyfunc(:Py_IncRef), Void, (PyPtr,), oargs[i])
    end
    ret = PyObject(ccall(pyfunc(:PyObject_CallObject), PyPtr,
                         (PyPtr,PyPtr), o, arg))
    if ret.o == C_NULL
        error("failure calling Python function")
    end
    jret = convert(returntype, ret)
    # TODO: check for errors
    return jret
end

#########################################################################

end # module PyCall
