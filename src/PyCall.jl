module PyCall

export pyinitialize, pyfinalize, pycall, pyimport, pyglobal, PyObject

import Base.convert
import Base.ref

const libpython = "libpython2.6" # fixme: don't hard-code this

typealias PyPtr Ptr{Void} # type for PythonObject* in ccall

#########################################################################

# initialize the Python interpreter (no-op on subsequent calls)
function pyinitialize()
    ccall((:Py_Initialize, libpython), Void, ())
end

# end the Python interpreter and free associated memory
function pyfinalize()
    ccall((:Py_Finalize, libpython), Void, ())
end

#########################################################################
# Wrapper around Python's C PyObject* type, with hooks to Python reference
# counting and conversion routines to/from C and Julia types.

type PyObject
    o::PyPtr
    function PyObject(o::PyPtr)
        po = new(o)
        finalizer(po, po -> ccall((:Py_DecRef, libpython), 
                                  Void, (PyPtr,), po.o))
        return po
    end
end

# conversion to pass PyObject as ccall arguments:
convert(::Type{PyPtr}, po::PyObject) = po.o

# conversions from Julia types to PyObject:

PyObject(i::Unsigned) = PyObject(ccall((:PyInt_FromSize_t, libpython),
                                       PyPtr, (Uint,), i))
PyObject(i::Integer) = PyObject(ccall((:PyInt_FromSsize_t, libpython),
                                      PyPtr, (Int,), i))

PyObject(b::Bool) = 
  PyObject(ccall((:PyBool_FromLong, libpython),
                 PyPtr, (OS_NAME == :Windows ? Int32 : Int,), b))

PyObject(r::Real) = PyObject(ccall((:PyFloat_FromDouble, libpython),
                                   PyPtr, (Float64,), r))

PyObject(c::Complex) = PyObject(ccall((:PyComplex_FromDoubles, libpython),
                                      PyPtr, (Float64,Float64), 
                                      real(c), imag(c)))

# fixme: PyString_* was renamed to PyBytes_* in Python 3.x?
PyObject(s::String) = PyObject(ccall((:PyString_FromString, libpython),
                                     PyPtr, (Ptr{Uint8},), bytestring(s)))

# conversions to Julia types from PyObject

convert{T<:Integer}(::Type{T}, po::PyObject) = 
  convert(T, ccall((:PyInt_AsSsize_t, libpython), Int, (PyPtr,), po))

convert(::Type{Bool}, po::PyObject) = 
  convert(Bool, ccall((:PyInt_AsSsize_t, libpython), Int, (PyPtr,), po))

convert{T<:Real}(::Type{T}, po::PyObject) = 
  convert(T, ccall((:PyFloat_AsDouble, libpython), Float64, (PyPtr,), po))

convert{T<:Complex}(::Type{T}, po::PyObject) = 
  convert(T, 
          complex128(ccall((:PyComplex_RealAsDouble, libpython), 
                           Float64, (PyPtr,), po),
                     ccall((:PyComplex_ImagAsDouble, libpython), 
                           Float64, (PyPtr,), po)))

convert(::Type{String}, po::PyObject) =
  bytestring(ccall((:PyString_AsString, libpython), 
                   Ptr{Uint8}, (PyPtr,), po))

#########################################################################
# For o::PyObject, make o["foo"] and o[:foo] equivalent to o.foo in Python

function ref(o::PyObject, s::String)
    p = ccall((:PyObject_GetAttrString, libpython), PyPtr,
              (PyPtr, Ptr{Uint8}), o, bytestring(s))
    if p == C_NULL
        throw(KeyError(s))
    end
    return PyObject(p)
end

ref(o::PyObject, s::Symbol) = ref(o, string(s))

#########################################################################

function pyimport(name::String)
    mod = PyObject(ccall((:PyImport_ImportModule, libpython), PyPtr,
                         (Ptr{Uint8},), bytestring(name)))
    # fixme: check for errors
    return mod
end

pyimport(name::Symbol) = pyimport(string(name))

# look up a global variable (in module __main__)
function pyglobal(name::String)
    # fixme: check for errors
    return PyObject(ccall((:PyObject_GetAttrString, libpython), PyPtr,
                          (PyPtr, Ptr{Uint8}), 
                          ccall((:PyImport_AddModule, libpython), PyPtr,
                                (Ptr{Uint8},), bytestring("__main__")),
                          bytestring(name)))
end

pyglobal(name::Symbol) = pyglobal(string(name))

#########################################################################

function pycall(o::PyObject, returntype::Type, args...)
    oargs = map(PyObject, args)
    # would rather call PyTuple_Pack, but calling varargs functions
    # with ccall and argument splicing seems problematic right now.
    arg = PyObject(ccall((:PyTuple_New, libpython), PyPtr, (Int,), 
                         length(args)))
    if arg.o == C_NULL
        error("failure creating Python argument tuple")
    end
    for i = 1:length(args)
        if 0 != ccall((:PyTuple_SetItem, libpython), Int32, (PyPtr,Int,PyPtr),
                      arg, i-1, oargs[i])
            error("error setting Python argument tuple")
        end
        # must incref since PyTuple_SetItem steals a reference
        ccall((:Py_IncRef, libpython), Void, (PyPtr,), oargs[i])
    end
    ret = PyObject(ccall((:PyObject_CallObject, libpython), PyPtr,
                         (PyPtr,PyPtr), o, arg))
    if ret.o == C_NULL
        error("failure calling Python function")
    end
    jret = convert(returntype, ret)
    # TODO: check for errors
    return jret
end

pycall(s::Union(String,Symbol), returntype::Type, args...) =
  pycall(pyglobal(s), returntype, args...)

#########################################################################

end # module PyCall
