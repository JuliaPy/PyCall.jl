# Conversions between Julia and Python types for the PyCall module.

#########################################################################
# Conversions of simple types (numbers and strings)

# conversions from Julia types to PyObject:

PyObject(i::Unsigned) = PyObject(@pycheckn ccall(pyfunc(:PyInt_FromSize_t),
                                                 PyPtr, (Uint,), i))
PyObject(i::Integer) = PyObject(@pycheckn ccall(pyfunc(:PyInt_FromSsize_t),
                                                PyPtr, (Int,), i))

PyObject(b::Bool) = PyObject(@pycheckn ccall(pyfunc(:PyBool_FromLong), 
                                             PyPtr, (Clong,), b))

PyObject(r::Real) = PyObject(@pycheckn ccall(pyfunc(:PyFloat_FromDouble),
                                             PyPtr, (Cdouble,), r))

PyObject(c::Complex) = PyObject(@pycheckn ccall(pyfunc(:PyComplex_FromDoubles),
                                                PyPtr, (Cdouble,Cdouble), 
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
  convert(T, @pycheck ccall(pyfunc(:PyFloat_AsDouble), Cdouble, (PyPtr,), po))

convert{T<:Complex}(::Type{T}, po::PyObject) = 
  convert(T,
    begin
        re = @pycheck ccall(pyfunc(:PyComplex_RealAsDouble),
                            Cdouble, (PyPtr,), po)
        complex128(re, ccall(pyfunc(:PyComplex_ImagAsDouble), 
                             Cdouble, (PyPtr,), po))
    end)

convert{T<:String}(::Type{T}, po::PyObject) =
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
        @pycheckzi ccall(pyfunc(:PyTuple_SetItem), Cint, (PyPtr,Int,PyPtr),
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
        @pycheckzi ccall(pyfunc(:PyList_SetItem), Cint, (PyPtr,Int,PyPtr),
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

# NumPy conversions (multidimensional arrays)
include("numpy.jl")

#########################################################################
# PyDict: no-copy wrapping of a Julia object around a Python dictionary

type PyDict{K,V} <: Associative{K,V}
    o::PyObject
    isdict::Bool # whether this is a Python Dict (vs. generic Mapping object)

    function PyDict(o::PyObject) 
        if o.o == C_NULL
            throw(ArgumentError("cannot make PyDict from NULL PyObject"))
        elseif pydict_query(o) == None
            throw(ArgumentError("only Dict and Mapping objects can be converted to PyDict"))
        end
        new(o, pyisinstance(o, :PyDict_Type))
    end
    function PyDict() 
        @pyinitialize
        new(PyObject(@pycheckni ccall(pyfunc(:PyDict_New), PyPtr, ())), true)
    end
end

PyDict(o::PyObject) = PyDict{PyAny,PyAny}(o)
PyDict() = PyDict{PyAny,PyAny}()
PyDict{K,V}(d::Dict{K,V}) = PyDict{K,V}(PyObject(d))
convert(::Type{PyDict}, o::PyObject) = PyDict(o)
convert{K,V}(::Type{PyDict{K,V}}, o::PyObject) = PyDict{K,V}(o)
convert(::Type{PyPtr}, d::PyDict) = d.o.o

has(d::PyDict, key) = 1 == ccall(pyfunc(d.isdict ? :PyDict_Contains :
                                                   :PyMapping_HasKey),
                                  Cint, (PyPtr, PyPtr), d, PyObject(key))

pyobject_call(d::PyDict, vec::String) = PyObject(@pycheckni ccall(pyfunc(:PyObject_CallMethod), PyPtr, (PyPtr,Ptr{Uint8},Ptr{Uint8}), d, bytestring(vec), C_NULL))

keys{T}(::Type{T}, d::PyDict) = convert(Vector{T}, d.isdict ? PyObject(@pycheckni ccall(pyfunc(:PyDict_Keys), PyPtr, (PyPtr,), d)) : pyobject_call(d, "keys"))

values{T}(::Type{T}, d::PyDict) = convert(Vector{T}, d.isdict ? PyObject(@pycheckni ccall(pyfunc(:PyDict_Values), PyPtr, (PyPtr,), d)) : pyobject_call(d, "values"))

similar{K,V}(d::PyDict{K,V}) = Dict{pyany_toany(K),pyany_toany(V)}()
eltype{K,V}(a::PyDict{K,V}) = (pyany_toany(K),pyany_toany(V))

function assign(d::PyDict, v, k)
    @pycheckzi ccall(pyfunc(:PyObject_SetItem), Cint, (PyPtr, PyPtr, PyPtr),
                     d, PyObject(k), PyObject(v))
    d
end

function get{K,V}(d::PyDict{K,V}, k, default)
    vo = ccall(pyfunc(:PyObject_GetItem), PyPtr, (PyPtr,PyPtr), d, PyObject(k))
    if vo == C_NULL
        pyerr_clear()
        return default
    else
        return convert(V, PyObject(vo))
    end
end

function delete!(d::PyDict, k)
    v = d[k]
    @pycheckzi ccall(pyfunc(d.isdict ? :PyDict_DelItem : :PyObject_DelItem),
                     Cint, (PyPtr, PyPtr), d, PyObject(k))
    return v
end

function delete!(d::PyDict, k, default)
    try
        return delete!(d, k)
    catch
        return default
    end
end

function empty!(d::PyDict)
    if d.isdict
        @pychecki ccall(pyfunc(:PyDict_Clear), Void, (PyPtr,), d)
    else
        # for generic Mapping items we must delete keys one by one
        for k in keys(d)
            delete!(d, k)
        end
    end
    return d
end

length(d::PyDict) = @pycheckz ccall(pyfunc(d.isdict ? :PyDict_Size
                                           :PyObject_Size), Int, (PyPtr,), d)
isempty(d::PyDict) = length(d) == 0

type PyDict_Iterator
    # arrays to pass key, value, and pos pointers to PyDict_Next
    ka::Array{PyPtr}
    va::Array{PyPtr}
    pa::Vector{Int}

    items::PyObject # items list, for generic Mapping objects

    i::Int # current position in items list (0-based)
    len::Int # length of items list
end

function start(d::PyDict)
    if d.isdict
        PyDict_Iterator(Array(PyPtr,1), Array(PyPtr,1), zeros(Int,1),
                        PyObject(C_NULL), 0, length(d))
    else
        items = convert(Vector{PyObject}, pyobject_call(d, PyObject, "items"))
        PyDict_Iterator(Array(PyPtr,0), Array(PyPtr,0), zeros(Int,0),
                        items, 0,
                        @pycheckz ccall(pyfunc(:PySequence_Size),
                                        Int, (PyPtr,), items))
    end
end

done(d::PyDict, itr::PyDict_Iterator) = itr.i >= itr.len

function next{K,V}(d::PyDict{K,V}, itr::PyDict_Iterator)
    if itr.items.o == C_NULL
        # Dict object, use PyDict_Next
        if 0 == ccall(pyfunc(:PyDict_Next), Cint,
                      (PyPtr, Ptr{Int}, Ptr{PyPtr}, Ptr{PyPtr}),
                      d, itr.pa, itr.ka, itr.va)
            error("unexpected end of PyDict_Next")
        end
        ko = pyincref(PyObject(itr.ka[1])) # PyDict_Next returns
        vo = pyincref(PyObject(itr.va[1])) #   borrowed ref, so incref
        ((convert(K,ko), convert(V,vo)),
         PyDict_Iterator(itr.ka, itr.va, itr.pa, itr.items, itr.i+1, itr.len))
    else
        # generic Mapping object, use items list
        (convert((K,V), PyObject(@pycheckni ccall(pyfunc(:PySequence_GetItem),
                                                  PyPtr, (PyPtr,Int), 
                                                  itr.items, itr.i))),
         PyDict_Iterator(itr.ka, itr.va, itr.pa, itr.items, itr.i+1, itr.len))
    end
end

function filter!(f::Function, d::PyDict)
    # We must use items(d) here rather than (k,v) in d,
    # because PyDict_Next does not permit changing the set of keys
    # during iteration.
    for (k,v) in items(d)
        if !f(k,v)
            delete!(d,k)
        end
    end
    return d
end

#########################################################################
# Dictionary conversions (copies)

function PyObject(d::Associative)
    o = PyObject(@pycheckn ccall(pyfunc(:PyDict_New), PyPtr, ()))
    for k in keys(d)
        @pycheckzi ccall(pyfunc(:PyDict_SetItem), Cint, (PyPtr,PyPtr,PyPtr),
                         o, PyObject(k), PyObject(d[k]))
    end
    return o
end

function convert{K,V}(::Type{Dict{K,V}}, o::PyObject)
    @pyinitialize
    copy(PyDict{K,V}(o))
end

#########################################################################
# Inferring Julia types at runtime from Python objects:
#
# [Note that we sometimes use the PyFoo_Check API and sometimes we use
#  PyObject_IsInstance(o, PyFoo_Type), since sometimes the former API
#  is a macro (hence inaccessible in Julia).]

# A type-query function f(o::PyObject) returns the Julia type
# for use with the convert function, or None if there isn't one.

pyint_query(o::PyObject) = pyisinstance(o, :PyInt_Type) ? 
  (pyisinstance(o, :PyBool_Type) ? Bool : Int) : None

pyfloat_query(o::PyObject) = pyisinstance(o, :PyFloat_Type) ? Float64 : None

pycomplex_query(o::PyObject) = 
  pyisinstance(o, :PyComplex_Type) ? Complex128 : None

pystring_query(o::PyObject) = pyisinstance(o, :PyString_Type) ? String : None

pyfunction_query(o::PyObject) = pyisinstance(o, :PyFunction_Type) || pyisinstance(o, BuiltinFunctionType) || pyisinstance(o, ufuncType) || pyisinstance(o, TypeType) || pyisinstance(o, MethodType) || pyisinstance(o, MethodWrapperType) ? Function : None

# we check for "items" attr since PyMapping_Check doesn't do this (it only
# checks for __getitem__) and PyMapping_Check returns true for some 
# scipy scalar array members, grrr.
pydict_query(o::PyObject) = pyisinstance(o, :PyDict_Type) || (pyquery(:PyMapping_Check, o) && ccall(pyfunc(:PyObject_HasAttrString), Cint, (PyPtr,Array{Uint8}), o, "items") == 1) ? Dict{PyAny,PyAny} : None

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
