# Conversions between Julia and Python types for the PyCall module.

#########################################################################
# Conversions of simple types (numbers and nothing)

# conversions from Julia types to PyObject:

PyObject(i::Unsigned) = PyObject(@pycheckn ccall(pyint_from_size_t::Ptr{Void},
                                                 PyPtr, (Uint,), i))
PyObject(i::Integer) = PyObject(@pycheckn ccall(pyint_from_ssize_t::Ptr{Void},
                                                PyPtr, (Int,), i))

PyObject(b::Bool) = PyObject(@pycheckn ccall((@pysym :PyBool_FromLong), 
                                             PyPtr, (Clong,), b))

PyObject(r::Real) = PyObject(@pycheckn ccall((@pysym :PyFloat_FromDouble),
                                             PyPtr, (Cdouble,), r))

PyObject(c::Complex) = PyObject(@pycheckn ccall((@pysym :PyComplex_FromDoubles),
                                                PyPtr, (Cdouble,Cdouble), 
                                                real(c), imag(c)))

PyObject(n::Nothing) = begin @pyinitialize; pyerr_check("PyObject(nothing)", pyincref((pynothing::PyObject).o)); end

# conversions to Julia types from PyObject

convert{T<:Integer}(::Type{T}, po::PyObject) = 
  convert(T, @pycheck ccall(pyint_as_ssize_t::Ptr{Void}, Int, (PyPtr,), po))

convert(::Type{Bool}, po::PyObject) = 
  convert(Bool, @pycheck ccall(pyint_as_ssize_t::Ptr{Void}, Int, (PyPtr,), po))

convert{T<:Real}(::Type{T}, po::PyObject) = 
  convert(T, @pycheck ccall((@pysym :PyFloat_AsDouble), Cdouble, (PyPtr,), po))

convert{T<:Complex}(::Type{T}, po::PyObject) = 
  convert(T,
    begin
        re = @pycheck ccall((@pysym :PyComplex_RealAsDouble),
                            Cdouble, (PyPtr,), po)
        complex128(re, ccall((@pysym :PyComplex_ImagAsDouble), 
                             Cdouble, (PyPtr,), po))
    end)

convert(::Type{Nothing}, po::PyObject) = nothing

#########################################################################
# String conversions (both bytes arrays and unicode strings)

PyObject(s::UTF8String) =
  PyObject(@pycheckn ccall(PyUnicode_DecodeUTF8::Ptr{Void},
                           PyPtr, (Ptr{Uint8}, Int, Ptr{Uint8}),
                           bytestring(s), length(s.data), C_NULL))

function PyObject(s::String)
    @pyinitialize
    if pyunicode_literals::Bool
        sb = bytestring(s)
        PyObject(@pycheckn ccall(PyUnicode_DecodeUTF8::Ptr{Void},
                                 PyPtr, (Ptr{Uint8}, Int, Ptr{Uint8}),
                                 sb, length(sb.data), C_NULL))
    else
        PyObject(@pycheckn ccall(pystring_fromstring::Ptr{Void},
                                 PyPtr, (Ptr{Uint8},), bytestring(s)))
    end
end

function convert{T<:String}(::Type{T}, po::PyObject)
    @pyinitialize
    if pyisinstance(po, @pysym :PyUnicode_Type)
        convert(T, PyObject(@pycheckni ccall(PyUnicode_AsUTF8String::Ptr{Void},
                                             PyPtr, (PyPtr,), po)))
    else
        convert(T, bytestring(@pycheckni ccall(pystring_asstring::Ptr{Void},
                                               Ptr{Uint8}, (PyPtr,), po)))
    end
end

# TODO: should symbols be converted to a subclass of Python strings/bytes,
#       so that PyAny conversion can convert it back to a Julia symbol?
PyObject(s::Symbol) = PyObject(string(s))
convert(::Type{Symbol}, po::PyObject) = symbol(convert(String, po))

#########################################################################
# Pointer conversions, using ctypes, PyCObject, or PyCapsule

PyObject(p::Ptr) = begin @pyinitialize; (py_void_p::Function)(p); end

function convert(::Type{Ptr{Void}}, po::PyObject)
    @pyinitialize
    if pyisinstance(po, c_void_p_Type::PyObject)
        v = po["value"]
        # ctypes stores the NULL pointer specially, grrr
        pynothing_query(v) == Nothing ? C_NULL : 
          convert(Ptr{Void}, convert(Uint, po["value"]))
    elseif pyisinstance(po, PyCObject_Type::Ptr{Void})
        @pychecki ccall((@pysym :PyCObject_AsVoidPtr), Ptr{Void}, (PyPtr,), po)
    elseif pyisinstance(po, PyCapsule_Type::Ptr{Void})
        @pychecki ccall((@pysym :PyCapsule_GetPointer), 
                        Ptr{Void}, (PyPtr,Ptr{Uint8}),
                        po, ccall((@pysym :PyCapsule_GetName), 
                                  Ptr{Uint8}, (PyPtr,), po))
    else
        convert(Ptr{Void}, convert(Uint, po))
    end
end

pyptr_query(po::PyObject) = pyisinstance(po, c_void_p_Type::PyObject) || pyisinstance(po, PyCObject_Type::Ptr{Void}) || pyisinstance(po, PyCapsule_Type::Ptr{Void}) ? Ptr{Void} : None

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
# Function conversion (see callback.jl for conversion the other way)

convert(::Type{Function}, po::PyObject) =
    function fn(args...; kwargs...)
        pycall(po, PyAny, args...; kwargs...)
    end

#########################################################################
# Tuple conversion

function PyObject(t::Tuple) 
    o = PyObject(@pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), 
                                 length(t)))
    for i = 1:length(t)
        oi = PyObject(t[i])
        @pycheckzi ccall((@pysym :PyTuple_SetItem), Cint, (PyPtr,Int,PyPtr),
                         o, i-1, oi)
        pyincref(oi) # PyTuple_SetItem steals the reference
    end
    return o
end

function convert(tt::NTuple{Type}, o::PyObject)
    len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
    if len != length(tt)
        throw(BoundsError())
    end
    ntuple(len, i ->
           convert(tt[i], PyObject(ccall((@pysym :PySequence_GetItem), PyPtr, 
                                         (PyPtr, Int), o, i-1))))
end

#########################################################################
# PyVector: no-copy wrapping of a Julia object around a Python sequence

type PyVector{T} <: AbstractVector{T}
    o::PyObject
    function PyVector(o::PyObject)
        if o.o == C_NULL
            throw(ArgumentError("cannot make PyVector from NULL PyObject"))
        elseif pysequence_query(o) == None
            throw(ArgumentError("only List and Sequence objects can be converted to PyVector"))
        end
        new(o)
    end
end

PyVector(o::PyObject) = PyVector{PyAny}(o)
convert(::Type{PyVector}, o::PyObject) = PyVector(o)
convert{T}(::Type{PyVector{T}}, o::PyObject) = PyVector{T}(o)
convert(::Type{PyPtr}, a::PyVector) = a.o.o

# when a PyVector is copied it is converted into an ordinary Julia Vector
similar(a::PyVector, T, dims::Dims) = Array(T, dims)
similar{T}(a::PyVector{T}) = similar(a, pyany_toany(T), size(a))
similar{T}(a::PyVector{T}, dims::Dims) = similar(a, pyany_toany(T), dims)
similar{T}(a::PyVector{T}, dims::Int...) = similar(a, pyany_toany(T), dims)
eltype{T}(::PyVector{T}) = pyany_toany(T)
eltype{T}(::Type{PyVector{T}}) = pyany_toany(T)

size(a::PyVector) = ((@pycheckzi ccall((@pysym :PySequence_Size), Int, (PyPtr,), a)),)

getindex(a::PyVector) = getindex(a, 1)
getindex{T}(a::PyVector{T}, i::Integer) = convert(T, PyObject(@pycheckni ccall((@pysym :PySequence_GetItem), PyPtr, (PyPtr, Int), a, i-1)))

setindex!(a::PyVector, v) = setindex!(a, v, 1)
setindex!(a::PyVector, v, i::Integer) = @pycheckzi ccall((@pysym :PySequence_SetItem), Cint, (PyPtr, Int, PyPtr), a, i-1, PyObject(v))

function delete!(a::PyVector, i::Integer)
    v = a[i]
    @pycheckzi ccall((@pysym :PySequence_DelItem), Cint, (PyPtr, Int), a, i-1)
    v
end

pop!(a::PyVector) = delete!(a, length(a))

summary{T}(a::PyVector{T}) = string(Base.dims2string(size(a)), " ",
                                   string(pyany_toany(T)), " PyVector")

#########################################################################
# Lists and 1d arrays.

# recursive conversion of A to a list of list of lists... starting
# with dimension dim and index i in A.
function array2py{T, N}(A::AbstractArray{T, N}, dim::Integer, i::Integer)
    if dim > N
        return PyObject(A[i])
    elseif dim == N # special case last dim to coarsen recursion leaves
        len = size(A, dim)
        s = N == 1 ? 1 : stride(A, dim)
        o = PyObject(@pycheckn ccall((@pysym :PyList_New), PyPtr, (Int,), len))
        for j = 0:len-1
            oi = PyObject(A[i+j*s])
            @pycheckzi ccall((@pysym :PyList_SetItem), Cint, (PyPtr,Int,PyPtr),
                             o, j, oi)
            pyincref(oi) # PyList_SetItem steals the reference  
        end
        return o
    else # dim < N: store multidimensional array as list of lists
        len = size(A, dim)
        s = stride(A, dim)
        o = PyObject(@pycheckn ccall((@pysym :PyList_New), PyPtr, (Int,), len))
        for j = 0:len-1
            oi = array2py(A, dim+1, i+j*s)
            @pycheckzi ccall((@pysym :PyList_SetItem), Cint, (PyPtr,Int,PyPtr),
                             o, j, oi)
            pyincref(oi) # PyList_SetItem steals the reference  
        end
        return o
    end
end

array2py(A::AbstractArray) = array2py(A, 1, 1)

PyObject(A::AbstractArray) = 
   ndims(A) <= 1 || method_exists(stride,(typeof(A),Int)) ? array2py(A) :
   pyjlwrap_new(A)

function py2array{TA,N}(T, A::Array{TA,N}, o::PyObject,
                        dim::Integer, i::Integer)
    if dim > N
        A[i] = convert(T, o)
        return A
    elseif dim == N
        len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        if len != size(A, dim)
            error("dimension mismatch in py2array")
        end
        s = stride(A, dim)
        for j = 0:len-1
            A[i+j*s] = convert(T, PyObject(ccall((@pysym :PySequence_GetItem),
                                                 PyPtr, (PyPtr, Int), o, j)))
        end
        return A
    else # dim < N: recursively extract list of lists into A
        len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        if len != size(A, dim)
            error("dimension mismatch in py2array")
        end
        s = stride(A, dim)
        for j = 0:len-1
            py2array(T, A, PyObject(ccall((@pysym :PySequence_GetItem),
                                       PyPtr, (PyPtr, Int), o, j)),
                     dim+1, i+j*s)
        end
        return A
    end
end

# figure out if we can treat o as a multidimensional array, and return
# the dimensions
function pyarray_dims(o::PyObject)
    len = ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
    if len < 0 || # not a sequence
       len+1 < 0 || # object pretending to be a sequence of infinite length
       pystring_query(o) != None ||
       pyisinstance(o, @pysym :PyTuple_Type)
        pyerr_clear()
        return ()
    elseif len == 0
        return (0,)
    end
    dims0 = pyarray_dims(PyObject(ccall((@pysym :PySequence_GetItem),
                                        PyPtr, (PyPtr, Int), o, 0)))
    if length(dims0) == 0 # not a nested sequence
        return (len,)
    end
    for j = 1:len-1
        dims = pyarray_dims(PyObject(ccall((@pysym :PySequence_GetItem),
                                           PyPtr, (PyPtr, Int), o, j)))
        if dims != dims0
            # elements don't have equal lengths, cannot
            # treat as multidimensional array
            return (len,)
        end
    end
    return tuple(len, dims0...)
end

function py2array(T, o::PyObject)
    dims = pyarray_dims(o)
    A = Array(pyany_toany(T), dims)
    py2array(T, A, o, 1, 1)
end
            
function convert{T}(::Type{Vector{T}}, o::PyObject) 
    len = ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
    if len < 0 || # not a sequence
       len+1 < 0  # object pretending to be a sequence of infinite length
        throw(ArgumentError("expected Python sequence"))
    end
    py2array(T, Array(pyany_toany(T), len), o, 1, 1)
end

convert(::Type{Array}, o::PyObject) = py2array(PyAny, o)

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
        new(o, pyisinstance(o, @pysym :PyDict_Type))
    end
    function PyDict() 
        @pyinitialize
        new(PyObject(@pycheckni ccall((@pysym :PyDict_New), PyPtr, ())), true)
    end
end

PyDict(o::PyObject) = PyDict{PyAny,PyAny}(o)
PyDict() = PyDict{PyAny,PyAny}()
PyDict{K,V}(d::Associative{K,V}) = PyDict{K,V}(PyObject(d))
convert(::Type{PyDict}, o::PyObject) = PyDict(o)
convert{K,V}(::Type{PyDict{K,V}}, o::PyObject) = PyDict{K,V}(o)
convert(::Type{PyPtr}, d::PyDict) = d.o.o

haskey(d::PyDict, key) = 1 == ccall(d.isdict ? (@pysym :PyDict_Contains)
                                          : (@pysym :PyMapping_HasKey),
                                  Cint, (PyPtr, PyPtr), d, PyObject(key))

pyobject_call(d::PyDict, vec::String) = PyObject(@pycheckni ccall((@pysym :PyObject_CallMethod), PyPtr, (PyPtr,Ptr{Uint8},Ptr{Uint8}), d, bytestring(vec), C_NULL))

keys{T}(::Type{T}, d::PyDict) = convert(Vector{T}, d.isdict ? PyObject(@pycheckni ccall((@pysym :PyDict_Keys), PyPtr, (PyPtr,), d)) : pyobject_call(d, "keys"))

values{T}(::Type{T}, d::PyDict) = convert(Vector{T}, d.isdict ? PyObject(@pycheckni ccall((@pysym :PyDict_Values), PyPtr, (PyPtr,), d)) : pyobject_call(d, "values"))

similar{K,V}(d::PyDict{K,V}) = Dict{pyany_toany(K),pyany_toany(V)}()
eltype{K,V}(a::PyDict{K,V}) = (pyany_toany(K),pyany_toany(V))

function setindex!(d::PyDict, v, k)
    @pycheckzi ccall((@pysym :PyObject_SetItem), Cint, (PyPtr, PyPtr, PyPtr),
                     d, PyObject(k), PyObject(v))
    d
end

function get{K,V}(d::PyDict{K,V}, k, default)
    vo = ccall((@pysym :PyObject_GetItem), PyPtr, (PyPtr,PyPtr), d, PyObject(k))
    if vo == C_NULL
        pyerr_clear()
        return default
    else
        return convert(V, PyObject(vo))
    end
end

function delete!(d::PyDict, k)
    v = d[k]
    @pycheckzi ccall(d.isdict ? (@pysym :PyDict_DelItem)
                              : (@pysym :PyObject_DelItem),
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
        @pychecki ccall((@pysym :PyDict_Clear), Void, (PyPtr,), d)
    else
        # for generic Mapping items we must delete keys one by one
        for k in keys(d)
            delete!(d, k)
        end
    end
    return d
end

length(d::PyDict) = @pycheckz ccall(d.isdict ? (@pysym :PyDict_Size)
                                             : (@pysym :PyObject_Size), 
                                    Int, (PyPtr,), d)
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
                        PyObject(), 0, length(d))
    else
        items = convert(Vector{PyObject}, pyobject_call(d, PyObject, "items"))
        PyDict_Iterator(Array(PyPtr,0), Array(PyPtr,0), zeros(Int,0),
                        items, 0,
                        @pycheckz ccall((@pysym :PySequence_Size),
                                        Int, (PyPtr,), items))
    end
end

done(d::PyDict, itr::PyDict_Iterator) = itr.i >= itr.len

function next{K,V}(d::PyDict{K,V}, itr::PyDict_Iterator)
    if itr.items.o == C_NULL
        # Dict object, use PyDict_Next
        if 0 == ccall((@pysym :PyDict_Next), Cint,
                      (PyPtr, Ptr{Int}, Ptr{PyPtr}, Ptr{PyPtr}),
                      d, itr.pa, itr.ka, itr.va)
            error("unexpected end of PyDict_Next")
        end
        ko = pyincref(itr.ka[1]) # PyDict_Next returns
        vo = pyincref(itr.va[1]) #   borrowed ref, so incref
        ((convert(K,ko), convert(V,vo)),
         PyDict_Iterator(itr.ka, itr.va, itr.pa, itr.items, itr.i+1, itr.len))
    else
        # generic Mapping object, use items list
        (convert((K,V), PyObject(@pycheckni ccall((@pysym :PySequence_GetItem),
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
    o = PyObject(@pycheckn ccall((@pysym :PyDict_New), PyPtr, ()))
    for k in keys(d)
        @pycheckzi ccall((@pysym :PyDict_SetItem), Cint, (PyPtr,PyPtr,PyPtr),
                         o, PyObject(k), PyObject(d[k]))
    end
    return o
end

function convert{K,V}(::Type{Dict{K,V}}, o::PyObject)
    @pyinitialize
    copy(PyDict{K,V}(o))
end

#########################################################################
# Ranges: integer ranges are converted to xrange,
#         while other ranges (<: AbstractVector) are converted to lists

xrange(start, stop, step) = pycall(pyxrange::PyObject, PyObject,
                                   start, stop, step)

function PyObject{T<:Integer}(r::Ranges{T})
    s = step(r)
    f = first(r)
    l = last(r) + s
    if max(f,l) > typemax(Clong) || min(f,l) < typemin(Clong)
        # in Python 2.x, xrange is limited to Clong
        PyObject(T[r...])
    else
        @pyinitialize
        xrange(f, l, s)
    end
end

function convert{T<:Ranges}(::Type{T}, o::PyObject)
    v = PyVector(o)
    len = length(v)
    if len == 0
        return 1:0 # no way to get more info from an xrange
    elseif len == 1
        start = v[1]
        return start:start
    else
        start = v[1]
        stop = v[len]
        step = v[2] - start
        return step == 1 ? (start:stop) : (start:step:stop)
    end
end

#########################################################################
# Inferring Julia types at runtime from Python objects:
#
# [Note that we sometimes use the PyFoo_Check API and sometimes we use
#  PyObject_IsInstance(o, PyFoo_Type), since sometimes the former API
#  is a macro (hence inaccessible in Julia).]

# A type-query function f(o::PyObject) returns the Julia type
# for use with the convert function, or None if there isn't one.

pyint_query(o::PyObject) = pyisinstance(o, pyint_type::Ptr{Void}) ? 
  (pyisinstance(o, @pysym :PyBool_Type) ? Bool : Int) : None

pyfloat_query(o::PyObject) = pyisinstance(o, @pysym :PyFloat_Type) ? Float64 : None

pycomplex_query(o::PyObject) = 
  pyisinstance(o, @pysym :PyComplex_Type) ? Complex128 : None

pystring_query(o::PyObject) = pyisinstance(o, pystring_type::Ptr{Void}) ? String : pyisinstance(o, @pysym :PyUnicode_Type) ? UTF8String : None

pyfunction_query(o::PyObject) = pyisinstance(o, @pysym :PyFunction_Type) || pyisinstance(o, BuiltinFunctionType::PyObject) || pyisinstance(o, ufuncType::PyObject) || pyisinstance(o, TypeType::PyObject) || pyisinstance(o, MethodType::PyObject) || pyisinstance(o, MethodWrapperType::PyObject) ? Function : None

pynothing_query(o::PyObject) = o.o == (pynothing::PyObject).o ? Nothing : None

# we check for "items" attr since PyMapping_Check doesn't do this (it only
# checks for __getitem__) and PyMapping_Check returns true for some 
# scipy scalar array members, grrr.
pydict_query(o::PyObject) = pyisinstance(o, @pysym :PyDict_Type) || (pyquery((@pysym :PyMapping_Check), o) && ccall((@pysym :PyObject_HasAttrString), Cint, (PyPtr,Array{Uint8}), o, "items") == 1) ? Dict{PyAny,PyAny} : None

function pysequence_query(o::PyObject)
    # pyquery(:PySequence_Check, o) always succeeds according to the docs,
    # but it seems we need to be careful; I've noticed that things like
    # scipy define "fake" sequence types with intmax lengths and other
    # problems
    if pyisinstance(o, @pysym :PyTuple_Type)
        len = @pycheckzi ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        return ntuple(len, i ->
                      pytype_query(PyObject(ccall((@pysym :PySequence_GetItem), 
                                                  PyPtr, (PyPtr,Int), o,i-1)),
                                   PyAny))
    elseif pyisinstance(o, pyxrange::PyObject)
        return Ranges
    else
        try
            otypestr = PyObject(@pycheckni ccall((@pysym :PyObject_GetItem), PyPtr, (PyPtr,PyPtr,), o["__array_interface__"], PyObject("typestr")))
            typestr = convert(String, otypestr)
            T = npy_typestrs[typestr[2:end]]
            return Array{T}
        catch
            # only handle PyList for now
            return pyisinstance(o, @pysym :PyList_Type) ? Array : None
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
    # Would be faster to have some kind of hash table here, but
    # that seems a bit tricky when we take subclasses into account
    @return_not_None pyint_query(o)
    @return_not_None pyfloat_query(o)
    @return_not_None pycomplex_query(o)
    @return_not_None pystring_query(o)
    @return_not_None pyfunction_query(o)
    @return_not_None pydict_query(o)
    @return_not_None pysequence_query(o)
    @return_not_None pyptr_query(o)
    @return_not_None pynothing_query(o)
    return default
end

pytype_query(o::PyObject) = pytype_query(o, PyObject)

function convert(::Type{PyAny}, o::PyObject)
    if (o.o == C_NULL)
        return o
    end
    try 
        T = pytype_query(o)
        if T == PyObject && is_pyjlwrap(o)
            return unsafe_pyjlwrap_to_objref(o.o)
        end
        convert(pytype_query(o), o)
    catch
        pyerr_clear() # just in case
        o
    end
end

#########################################################################
