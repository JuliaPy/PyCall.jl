# Conversions between Julia and Python types for the PyCall module.

#########################################################################
# Conversions of simple types (numbers and strings)

# conversions from Julia types to PyObject:

PyObject(i::Unsigned) = PyObject(@pycheckn ccall(pyfunc(:PyInt_FromSize_t),
                                                 PyPtr, (Uint,), i))
PyObject(i::Integer) = PyObject(@pycheckn ccall(pyfunc(:PyInt_FromSsize_t),
                                                PyPtr, (Int,), i))

PyObject(b::Bool) = OS_NAME == :Windows ?
  PyObject(@pycheckn ccall(pyfunc(:PyBool_FromLong), PyPtr, (Int32,), b)) :
  PyObject(@pycheckn ccall(pyfunc(:PyBool_FromLong), PyPtr, (Int,), b))

PyObject(r::Real) = PyObject(@pycheckn ccall(pyfunc(:PyFloat_FromDouble),
                                             PyPtr, (Float64,), r))

PyObject(c::Complex) = PyObject(@pycheckn ccall(pyfunc(:PyComplex_FromDoubles),
                                                PyPtr, (Float64,Float64), 
                                                real(c), imag(c)))

# fixme: PyString_* was renamed to PyBytes_* in Python 3.x?
PyObject(s::String) = PyObject(@pycheckn ccall(pyfunc(:PyString_FromString),
                                               PyPtr, (Ptr{Uint8},),
                                               bytestring(s)))

# conversions to Julia types from PyObject

convert{T<:Integer}(::Type{T}, po::PyObject) = 
  convert(T, @pycheck ccall(pyfunc(:PyInt_AsSsize_t), Int, (PyPtr,), po))

convert(::Type{Bool}, po::PyObject) = 
  convert(Bool, @pycheck ccall(pyfunc(:PyInt_AsSsize_t), Int, (PyPtr,), po))

convert{T<:Real}(::Type{T}, po::PyObject) = 
  convert(T, @pycheck ccall(pyfunc(:PyFloat_AsDouble), Float64, (PyPtr,), po))

convert{T<:Complex}(::Type{T}, po::PyObject) = 
  convert(T,
    begin
        re = @pycheck ccall(pyfunc(:PyComplex_RealAsDouble),
                            Float64, (PyPtr,), po)
        complex128(re, ccall(pyfunc(:PyComplex_ImagAsDouble), 
                             Float64, (PyPtr,), po))
    end)

convert(::Type{String}, po::PyObject) =
  bytestring(@pycheck ccall(pyfunc(:PyString_AsString),
                             Ptr{Uint8}, (PyPtr,), po))

#########################################################################
# for automatic conversions, I pass Vector{PyAny}, NTuple{PyAny}, etc.,
# but since PyAny is an abstract type I need to convert this to Any
# before actually creating the Julia object

# I want to use a union, but this seems to confuse Julia's method
# dispatch for the convert function in some circumstances
# typealias PyAny Union(PyObject, Int, Bool, Float64, Complex128, String, Function, Dict, Tuple, Array)
abstract PyAny

# I originally implemented this via multiple dispatch, with
#   pyany_toany(::Type{PyAny}), pyany_toany(x::Tuple), and pyany_toany(x),
# but Julia seemed to get easily confused about which one to call.
pyany_toany(x) = isa(x, Type{PyAny}) ? Any : (isa(x, Tuple) ? 
                                              map(pyany_toany, x) : x)

# no-op conversions
for T in (:PyObject, :Int, :Bool, :Float64, :Complex128, :String, 
          :Function, :Dict, :Tuple, :Array)
    @eval convert(::Type{PyAny}, x::$T) = x
end

#########################################################################
# Function conversion (TODO: Julia to Python conversion for callbacks)

convert(::Type{Function}, po::PyObject) =
    (args...) -> pycall(po, PyAny, args...)

#########################################################################
# Tuple conversion

function PyObject(t::Tuple) 
    o = PyObject(@pycheckn ccall(pyfunc(:PyTuple_New), PyPtr, (Int,), 
                                 length(t)))
    for i = 1:length(t)
        oi = PyObject(t[i])
        @pycheckzi ccall(pyfunc(:PyTuple_SetItem), Int32, (PyPtr,Int,PyPtr),
                         o, i-1, oi)
        pyincref(oi) # PyTuple_SetItem steals the reference
    end
    return o
end

function convert(tt::NTuple{Type}, o::PyObject)
    len = @pycheckz ccall(pyfunc(:PySequence_Size), Int, (PyPtr,), o)
    if len != length(tt)
        throw(BoundsError())
    end
    ntuple(len, i ->
           convert(tt[i], PyObject(ccall(pyfunc(:PySequence_GetItem), PyPtr, 
                                         (PyPtr, Int), o, i-1))))
end

#########################################################################
# Lists and 1d arrays.

function PyObject(v::AbstractVector)
    o = PyObject(@pycheckn ccall(pyfunc(:PyList_New), PyPtr,(Int,), length(v)))
    for i = 1:length(v)
        oi = PyObject(v[i])
        @pycheckzi ccall(pyfunc(:PyList_SetItem), Int32, (PyPtr,Int,PyPtr),
                         o, i-1, oi)
        pyincref(oi) # PyList_SetItem steals the reference
    end
    return o
end

function convert{T}(::Type{Vector{T}}, o::PyObject)
    len = @pycheckz ccall(pyfunc(:PySequence_Size), Int, (PyPtr,), o)
    if len < 0 || len+1 < 0 
        # scipy IndexExpression instances pretend to be sequences
        # with infinite (intmax) length, so we need to catch this, grr
        throw(ArgumentError("invalid PySequence length $len"))
    end
    TA = pyany_toany(T)
    TA[ convert(T, PyObject(ccall(pyfunc(:PySequence_GetItem), PyPtr, 
                                  (PyPtr, Int), o, i-1))) for i in 1:len ]
end

#########################################################################
# Dictionaries (TODO: no-copy conversion?)

function PyObject(d::Associative)
    o = PyObject(@pycheckn ccall(pyfunc(:PyDict_New), PyPtr, ()))
    for k in keys(d)
        @pycheckzi ccall(pyfunc(:PyDict_SetItem), Int32, (PyPtr,PyPtr,PyPtr),
                         o, PyObject(k), PyObject(d[k]))
    end
    return o
end

function convert{K,V}(::Type{Dict{K,V}}, o::PyObject)
    KA = pyany_toany(K)
    VA = pyany_toany(V)
    d = Dict{KA,VA}()
    # arrays to pass key, value, and pos pointers to PyDict_Next
    ka = Array(PyPtr, 1)
    va = Array(PyPtr, 1)
    pa = zeros(Int, 1) # must be initialized to zero
    @pyinitialize
    if pyisinstance(o, :PyDict_Type)
        # Dict loop is more efficient than items copy needed for Mapping below
        while 0 != ccall(pyfunc(:PyDict_Next), Int32, 
                         (PyPtr, Ptr{Int}, Ptr{PyPtr}, Ptr{PyPtr}),
                         o, pa, ka, va)
            ko = pyincref(PyObject(ka[1])) # PyDict_Next returns
            vo = pyincref(PyObject(va[1])) #   borrowed ref, so incref
            merge!(d, (KA=>VA)[convert(K,ko) => convert(V,vo)])
        end
    elseif pyquery(:PyMapping_Check, o)
        # use generic Python mapping protocol
        items = convert(Vector{(PyObject,PyObject)},
                        pycall(o["items"], PyObject))
        for (ko,vo) in items
            merge!(d, (KA=>VA)[convert(K, ko) => convert(V, vo)])
        end
    else
        throw(ArgumentError("only Mapping objects can be converted to Dict"))
    end
    return d
end

#########################################################################
# NumPy conversions (multidimensional arrays)

include("numpy.jl")

#########################################################################
# Inferring Julia types at runtime from Python objects.
#
# Note that we sometimes use the PyFoo_Check API and sometimes we use
# PyObject_IsInstance(o, PyFoo_Type), since sometimes the former API
# is a macro (hence inaccessible in Julia).

# A type-query function f(o::PyObject) returns the Julia type
# for use with the convert function, or None if there isn't one.

pyint_query(o::PyObject) = pyisinstance(o, :PyInt_Type) ? 
  (pyisinstance(o, :PyBool_Type) ? Bool : Int) : None

pyfloat_query(o::PyObject) = pyisinstance(o, :PyFloat_Type) ? Float64 : None

pycomplex_query(o::PyObject) = 
  pyisinstance(o, :PyComplex_Type) ? Complex128 : None

pystring_query(o::PyObject) = pyisinstance(o, :PyString_Type) ? String : None

pyfunction_query(o::PyObject) = pyisinstance(o, :PyFunction_Type) || pyisinstance(o, BuiltinFunctionType) || pyisinstance(o, ufuncType) ? Function : None

# we check for "items" attr since PyMapping_Check doesn't do this (it only
# checks for __getitem__) and PyMapping_Check returns true for some 
# scipy scalar array members, grrr.
pydict_query(o::PyObject) = pyisinstance(o, :PyDict_Type) || (pyquery(:PyMapping_Check, o) && ccall(pyfunc(:PyObject_HasAttrString), Int32, (PyPtr,Array{Uint8}), o, "items") == 1) ? Dict{PyAny,PyAny} : None

function pysequence_query(o::PyObject)
    # pyquery(:PySequence_Check, o) always succeeds according to the docs,
    # but it seems we need to be careful; I've noticed that things like
    # scipy define "fake" sequence types with intmax lengths and other
    # problems
    if pyisinstance(o, :PyTuple_Type)
        len = @pycheckzi ccall(pyfunc(:PySequence_Size), Int, (PyPtr,), o)
        return ntuple(len, i ->
                      pytype_query(PyObject(ccall(pyfunc(:PySequence_GetItem), 
                                                  PyPtr, (PyPtr,Int), o,i-1)),
                                   PyAny))
    else
        try
            otypestr = PyObject(@pycheckni ccall(pyfunc(:PyObject_GetItem), PyPtr, (PyPtr,PyPtr,), o["__array_interface__"], PyObject("typestr")))
            typestr = convert(String, otypestr)
            T = npy_typestrs[typestr[2:end]]
            return Array{T}
        catch
            # only handle PyList for now
            return pyisinstance(o, :PyList_Type) ? Vector{PyAny} : None
        end
    end
end

macro return_not_None(ex)
    quote
        T = $ex
        if T != None
            return T
        end
    end
end

function pytype_query(o::PyObject, default::Type)
    @pyinitialize
    @return_not_None pyint_query(o)
    @return_not_None pyfloat_query(o)
    @return_not_None pycomplex_query(o)
    @return_not_None pystring_query(o)
    @return_not_None pyfunction_query(o)
    @return_not_None pydict_query(o)
    @return_not_None pysequence_query(o)
    return default
end

pytype_query(o::PyObject) = pytype_query(o, PyObject)

function convert(::Type{PyAny}, o::PyObject)
    try 
        convert(pytype_query(o), o)
    catch
        pyerr_clear() # just in case
        o
    end
end

#########################################################################
