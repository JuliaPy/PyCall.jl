# Conversions between Julia and Python types for the PyCall module.

#########################################################################
# Conversions of simple types (numbers and nothing)

# conversions from Julia types to PyObject:

PyObject(i::Unsigned) = PyObject(@pycheckn ccall(@pysym(PyInt_FromSize_t),
                                                 PyPtr, (UInt,), i))
PyObject(i::Integer) = PyObject(@pycheckn ccall(@pysym(PyInt_FromSsize_t),
                                                PyPtr, (Int,), i))

PyObject(b::Bool) = PyObject(@pycheckn ccall((@pysym :PyBool_FromLong),
                                             PyPtr, (Clong,), b))

PyObject(r::Real) = PyObject(@pycheckn ccall((@pysym :PyFloat_FromDouble),
                                             PyPtr, (Cdouble,), r))

PyObject(c::Complex) = PyObject(@pycheckn ccall((@pysym :PyComplex_FromDoubles),
                                                PyPtr, (Cdouble,Cdouble),
                                                real(c), imag(c)))

PyObject(n::Void) = pyerr_check("PyObject(nothing)", pyincref(pynothing))

# conversions to Julia types from PyObject

# Numpy scalars need to be converted to ordinary Python scalars with
# the item() method before passing to the Python API conversion functions
asscalar(o::PyObject) = pyisinstance(o, npy_number) ? pycall(o["item"], PyObject) : o

convert{T<:Integer}(::Type{T}, po::PyObject) =
  convert(T, @pycheck ccall(@pysym(PyInt_AsSsize_t), Int, (PyPtr,), asscalar(po)))

if WORD_SIZE == 32
  convert{T<:Union{Int64,UInt64}}(::Type{T}, po::PyObject) =
    @pycheck ccall((@pysym :PyLong_AsLongLong), T, (PyPtr,), asscalar(po))
end

convert(::Type{Bool}, po::PyObject) =
    0 != @pycheck ccall(@pysym(PyInt_AsSsize_t), Int, (PyPtr,), asscalar(po))

convert{T<:Real}(::Type{T}, po::PyObject) =
  convert(T, @pycheck ccall((@pysym :PyFloat_AsDouble), Cdouble, (PyPtr,), asscalar(po)))

function convert{T<:Complex}(::Type{T}, po_::PyObject)
    po = asscalar(po_)
    convert(T,
            begin
                re = @pycheck ccall((@pysym :PyComplex_RealAsDouble),
                                    Cdouble, (PyPtr,), po)
                complex(re, ccall((@pysym :PyComplex_ImagAsDouble),
                                  Cdouble, (PyPtr,), po))
            end)
end

convert(::Type{Void}, po::PyObject) = nothing

#########################################################################
# String conversions (both bytes arrays and unicode strings)

PyObject(s::UTF8String) =
  PyObject(@pycheckn ccall(@pysym(PyUnicode_DecodeUTF8),
                           PyPtr, (Ptr{UInt8}, Int, Ptr{UInt8}),
                           s, sizeof(s), C_NULL))

function PyObject(s::AbstractString)
    sb = bytestring(s)
    if pyunicode_literals
        PyObject(@pycheckn ccall(@pysym(PyUnicode_DecodeUTF8),
                                 PyPtr, (Ptr{UInt8}, Int, Ptr{UInt8}),
                                 sb, sizeof(sb), C_NULL))
    else
        PyObject(@pycheckn ccall(@pysym(PyString_FromStringAndSize),
                                 PyPtr, (Ptr{UInt8}, Int), sb, sizeof(sb)))
    end
end

const _ps_ptr= Ptr{UInt8}[C_NULL]
const _ps_len = Int[0]
function convert{T<:AbstractString}(::Type{T}, po::PyObject)
    if pyisinstance(po, @pyglobalobj :PyUnicode_Type)
        convert(T, PyObject(@pycheckn ccall(@pysym(PyUnicode_AsUTF8String),
                                             PyPtr, (PyPtr,), po)))
    else
        @pycheckz ccall(@pysym(PyString_AsStringAndSize),
                        Cint, (PyPtr, Ptr{Ptr{UInt8}}, Ptr{Int}),
                        po, _ps_ptr, _ps_len)
        convert(T, bytestring(_ps_ptr[1], _ps_len[1]))
    end
end

# TODO: should symbols be converted to a subclass of Python strings/bytes,
#       so that PyAny conversion can convert it back to a Julia symbol?
PyObject(s::Symbol) = PyObject(string(s))
convert(::Type{Symbol}, po::PyObject) = symbol(convert(AbstractString, po))

#########################################################################
# ByteArray conversions

PyObject(a::Vector{UInt8}) =
  PyObject(@pycheckn ccall((@pysym :PyByteArray_FromStringAndSize),
                           PyPtr, (Ptr{UInt8}, Int), a, length(a)))

ispybytearray(po::PyObject) =
  pyisinstance(po, @pyglobalobj :PyByteArray_Type)

function convert(::Type{Vector{UInt8}}, po::PyObject)
    b = PyBuffer(po)
    iscontiguous(b) || error("a contiguous buffer is required")
    return copy(pointer_to_array(Ptr{UInt8}(pointer(b)), sizeof(b)))
end

# TODO: support zero-copy PyByteArray <: AbstractVector{UInt8} object

#########################################################################
# Pointer conversions, using ctypes or PyCapsule

PyObject(p::Ptr) = py_void_p(p)

function convert(::Type{Ptr{Void}}, po::PyObject)
    if pyisinstance(po, c_void_p_Type)
        v = po["value"]
        # ctypes stores the NULL pointer specially, grrr
        pynothing_query(v) == Void ? C_NULL :
          convert(Ptr{Void}, convert(UInt, po["value"]))
    elseif pyisinstance(po, @pyglobalobj(:PyCapsule_Type))
        @pycheck ccall((@pysym :PyCapsule_GetPointer),
                       Ptr{Void}, (PyPtr,Ptr{UInt8}),
                       po, ccall((@pysym :PyCapsule_GetName),
                                 Ptr{UInt8}, (PyPtr,), po))
    else
        convert(Ptr{Void}, convert(UInt, po))
    end
end

pyptr_query(po::PyObject) = pyisinstance(po, c_void_p_Type) || pyisinstance(po, @pyglobalobj(:PyCapsule_Type)) ? Ptr{Void} : Union{}

#########################################################################
# for automatic conversions, I pass Vector{PyAny}, NTuple{N, PyAny}, etc.,
# but since PyAny is an abstract type I need to convert this to Any
# before actually creating the Julia object

# I want to use a union, but this seems to confuse Julia's method
# dispatch for the convert function in some circumstances
# typealias PyAny Union{PyObject, Int, Bool, Float64, Complex128, AbstractString, Function, Dict, Tuple, Array}
abstract PyAny

pyany_toany(T::Type) = T
pyany_toany(T::Type{PyAny}) = Any
pyany_toany(T::Type{Vararg{PyAny}}) = Vararg{Any}
pyany_toany{T<:Tuple}(t::Type{T}) = Tuple{map(pyany_toany, t.types)...}

# PyAny acts like Any for conversions, except for converting PyObject (below)
convert(::Type{PyAny}, x) = x

#########################################################################
# Function conversion (see callback.jl for conversion the other way)
# (rarely needed given call overloading in Julia 0.4)

convert(::Type{Function}, po::PyObject) =
    function fn(args...; kwargs...)
        pycall(po, PyAny, args...; kwargs...)
    end

#########################################################################
# Tuple conversion.  Julia Pairs are treated as Python tuples.

function PyObject(t::Union{Tuple,Pair})
    len = endof(t) # endof, not length, because of julia#14924
    o = PyObject(@pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), len))
    for i = 1:len
        oi = PyObject(t[i])
        @pycheckz ccall((@pysym :PyTuple_SetItem), Cint, (PyPtr,Int,PyPtr),
                         o, i-1, oi)
        pyincref(oi) # PyTuple_SetItem steals the reference
    end
    return o
end

function convert{T<:Tuple}(tt::Type{T}, o::PyObject)
    len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
    if len != length(tt.types)
        throw(BoundsError())
    end
    ntuple((i ->
            convert(tt.types[i],
                    PyObject(ccall((@pysym :PySequence_GetItem), PyPtr,
                                   (PyPtr, Int), o, i-1)))),
           len)
end

function convert{K,V}(::Type{Pair{K,V}}, o::PyObject)
    k, v = convert(Tuple{K,V}, o)
    return Pair(k, v)
end

#########################################################################
# PyVector: no-copy wrapping of a Julia object around a Python sequence

"""
    PyVector(o::PyObject)

This returns a PyVector object, which is a wrapper around an arbitrary Python list or sequence object. 

Alternatively, `PyVector` can be used as the return type for a `pycall` that returns a sequence object (including tuples).
"""
type PyVector{T} <: AbstractVector{T}
    o::PyObject
    function PyVector(o::PyObject)
        if o.o == C_NULL
            throw(ArgumentError("cannot make PyVector from NULL PyObject"))
        end
        new(o)
    end
end

PyVector(o::PyObject) = PyVector{PyAny}(o)
PyObject(a::PyVector) = a.o
convert(::Type{PyVector}, o::PyObject) = PyVector(o)
convert{T}(::Type{PyVector{T}}, o::PyObject) = PyVector{T}(o)
unsafe_convert(::Type{PyPtr}, a::PyVector) = a.o.o
PyVector(a::PyVector) = a
PyVector{T}(a::AbstractVector{T}) = PyVector{T}(array2py(a))

# when a PyVector is copied it is converted into an ordinary Julia Vector
similar(a::PyVector, T, dims::Dims) = Array(T, dims)
similar{T}(a::PyVector{T}) = similar(a, pyany_toany(T), size(a))
similar{T}(a::PyVector{T}, dims::Dims) = similar(a, pyany_toany(T), dims)
similar{T}(a::PyVector{T}, dims::Int...) = similar(a, pyany_toany(T), dims)
eltype{T}(::PyVector{T}) = pyany_toany(T)
eltype{T}(::Type{PyVector{T}}) = pyany_toany(T)

size(a::PyVector) = (length(a.o),)

getindex(a::PyVector) = getindex(a, 1)
getindex{T}(a::PyVector{T}, i::Integer) = convert(T, PyObject(@pycheckn ccall((@pysym :PySequence_GetItem), PyPtr, (PyPtr, Int), a, i-1)))

setindex!(a::PyVector, v) = setindex!(a, v, 1)
function setindex!(a::PyVector, v, i::Integer)
    @pycheckz ccall((@pysym :PySequence_SetItem), Cint, (PyPtr, Int, PyPtr), a, i-1, PyObject(v))
    v
end

summary{T}(a::PyVector{T}) = string(Base.dims2string(size(a)), " ",
                                   string(pyany_toany(T)), " PyVector")

splice!(a::PyVector, i::Integer) = splice!(a.o, i)
function splice!{T,I<:Integer}(a::PyVector{T}, indices::AbstractVector{I})
    v = pyany_toany(T)[a[i] for i in indices]
    for i in sort(indices, rev=true)
        @pycheckz ccall((@pysym :PySequence_DelItem), Cint, (PyPtr, Int), a, i-1)
    end
    v
end
pop!(a::PyVector) = pop!(a.o)
shift!(a::PyVector) = shift!(a.o)
empty!(a::PyVector) = empty!(a.o)

# only works for List subtypes:
push!(a::PyVector, item) = push!(a.o, item)
insert!(a::PyVector, i::Integer, item) = insert!(a.o, i, item)
unshift!(a::PyVector, item) = unshift!(a.o, item)
prepend!(a::PyVector, items) = prepend!(a.o, items)
append!{T}(a::PyVector{T}, items) = PyVector{T}(append!(a.o, items))

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
            @pycheckz ccall((@pysym :PyList_SetItem), Cint, (PyPtr,Int,PyPtr),
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
            @pycheckz ccall((@pysym :PyList_SetItem), Cint, (PyPtr,Int,PyPtr),
                             o, j, oi)
            pyincref(oi) # PyList_SetItem steals the reference
        end
        return o
    end
end

array2py(A::AbstractArray) = array2py(A, 1, 1)

PyObject(A::AbstractArray) =
   ndims(A) <= 1 || method_exists(stride, Tuple{typeof(A),Int}) ? array2py(A) :
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
function pyarray_dims(o::PyObject, forcelist=true)
    if !(forcelist || pyisinstance(o, @pyglobalobj :PyList_Type))
        return () # too many non-List types can pretend to be sequences
    end
    len = ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
    if len == 0
        return (0,)
    end
    dims0 = pyarray_dims(PyObject(ccall((@pysym :PySequence_GetItem),
                                        PyPtr, (PyPtr, Int), o, 0)),
                         false)
    if isempty(dims0) # not a nested sequence
        return (len,)
    end
    for j = 1:len-1
        dims = pyarray_dims(PyObject(ccall((@pysym :PySequence_GetItem),
                                           PyPtr, (PyPtr, Int), o, j)),
                            false)
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
        pyerr_clear()
        throw(ArgumentError("expected Python sequence"))
    end
    py2array(T, Array(pyany_toany(T), len), o, 1, 1)
end

convert(::Type{Array}, o::PyObject) = py2array(PyAny, o)
convert{T}(::Type{Array{T}}, o::PyObject) = py2array(T, o)

# NumPy conversions (multidimensional arrays)
include("numpy.jl")

#########################################################################
# PyDict: no-copy wrapping of a Julia object around a Python dictionary

"""
    PyDict(o::PyObject)
    PyDict(d::Dict{K,V})

This returns a PyDict, which is a no-copy wrapper around a Python dictionary.

Alternatively, you can specify the return type of a `pycall` as PyDict. 
"""
type PyDict{K,V} <: Associative{K,V}
    o::PyObject
    isdict::Bool # whether this is a Python Dict (vs. generic Mapping object)

    function PyDict(o::PyObject)
        if o.o == C_NULL
            throw(ArgumentError("cannot make PyDict from NULL PyObject"))
        elseif pydict_query(o) == Union{}
            throw(ArgumentError("only Dict and Mapping objects can be converted to PyDict"))
        end
        new(o, pyisinstance(o, @pyglobalobj :PyDict_Type))
    end
    function PyDict()
        new(PyObject(@pycheckn ccall((@pysym :PyDict_New), PyPtr, ())), true)
    end
end

PyDict(o::PyObject) = PyDict{PyAny,PyAny}(o)
PyObject(d::PyDict) = d.o
PyDict() = PyDict{PyAny,PyAny}()
PyDict{K,V}(d::Associative{K,V}) = PyDict{K,V}(PyObject(d))
PyDict(d::Associative{Any,Any}) = PyDict{PyAny,PyAny}(PyObject(d))
PyDict{V}(d::Associative{Any,V}) = PyDict{PyAny,V}(PyObject(d))
PyDict{K}(d::Associative{K,Any}) = PyDict{K,PyAny}(PyObject(d))
convert(::Type{PyDict}, o::PyObject) = PyDict(o)
convert{K,V}(::Type{PyDict{K,V}}, o::PyObject) = PyDict{K,V}(o)
unsafe_convert(::Type{PyPtr}, d::PyDict) = d.o.o

haskey(d::PyDict, key) = 1 == (d.isdict ?
                               ccall(@pysym(:PyDict_Contains), Cint, (PyPtr, PyPtr), d, PyObject(key)) :
                               ccall(@pysym(:PyMapping_HasKey), Cint, (PyPtr, PyPtr), d, PyObject(key)))

keys{T}(::Type{T}, d::PyDict) = convert(Vector{T}, d.isdict ? PyObject(@pycheckn ccall((@pysym :PyDict_Keys), PyPtr, (PyPtr,), d)) : pycall(d.o["keys"], PyObject))

values{T}(::Type{T}, d::PyDict) = convert(Vector{T}, d.isdict ? PyObject(@pycheckn ccall((@pysym :PyDict_Values), PyPtr, (PyPtr,), d)) : pycall(d.o["values"], PyObject))

similar{K,V}(d::PyDict{K,V}) = Dict{pyany_toany(K),pyany_toany(V)}()
eltype{K,V}(a::PyDict{K,V}) = Pair{pyany_toany(K),pyany_toany(V)}
Base.keytype{K,V}(::PyDict{K,V}) = pyany_toany(K)
Base.valtype{K,V}(::PyDict{K,V}) = pyany_toany(V)
Base.keytype{K,V}(::Type{PyDict{K,V}}) = pyany_toany(K)
Base.valtype{K,V}(::Type{PyDict{K,V}}) = pyany_toany(V)

function setindex!(d::PyDict, v, k)
    @pycheckz ccall((@pysym :PyObject_SetItem), Cint, (PyPtr, PyPtr, PyPtr),
                     d, PyObject(k), PyObject(v))
    v
end

get{K,V}(d::PyDict{K,V}, k, default) = get(d.o, V, k, default)

function pop!(d::PyDict, k)
    v = d[k]
    @pycheckz (d.isdict ? ccall(@pysym(:PyDict_DelItem), Cint, (PyPtr, PyPtr), d, PyObject(k))
                : ccall(@pysym(:PyObject_DelItem), Cint, (PyPtr, PyPtr), d, PyObject(k)))
    return v
end

function pop!(d::PyDict, k, default)
    try
        return pop!(d, k)
    catch
        return default
    end
end

function delete!(d::PyDict, k)
    e = (d.isdict ? ccall(@pysym(:PyDict_DelItem), Cint, (PyPtr, PyPtr), d, PyObject(k))
         : ccall(@pysym(:PyObject_DelItem), Cint, (PyPtr, PyPtr), d, PyObject(k)))
    if e == -1
        pyerr_clear() # delete! ignores errors in Julia
    end
    return d
end

function empty!(d::PyDict)
    if d.isdict
        @pycheck ccall((@pysym :PyDict_Clear), Void, (PyPtr,), d)
    else
        # for generic Mapping items we must delete keys one by one
        for k in keys(d)
            delete!(d, k)
        end
    end
    return d
end

length(d::PyDict) = @pycheckz (d.isdict ? ccall(@pysym(:PyDict_Size), Int, (PyPtr,), d)
                               : ccall(@pysym(:PyObject_Size), Int, (PyPtr,), d))
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
                        PyNULL(), 0, length(d))
    else
        items = convert(Vector{PyObject}, pycall(d.o["items"], PyObject))
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
        (Pair(convert(K,ko), convert(V,vo)),
         PyDict_Iterator(itr.ka, itr.va, itr.pa, itr.items, itr.i+1, itr.len))
    else
        # generic Mapping object, use items list
        (convert(Pair{K,V}, PyObject(@pycheckn ccall((@pysym :PySequence_GetItem),
                                      PyPtr, (PyPtr,Int), itr.items, itr.i))),
         PyDict_Iterator(itr.ka, itr.va, itr.pa, itr.items, itr.i+1, itr.len))
    end
end

if VERSION < v"0.5.0-dev+9920" # julia PR #14937
    # We can't use the Base.filter! implementation because it worked
    # by `for (k,v) in d; !f(k,v) && delete!(d,k); end`, but the PyDict_Next
    # iterator function in Python is explicitly documented to say that
    # you shouldn't modify the dictionary during iteration.
    function filter!(f::Function, d::PyDict)
        badkeys = Array(keytype(d), 0)
        for (k,v) in d
            f(k,v) || push!(badkeys, k)
        end
        for k in badkeys
            delete!(d, k)
        end
        return d
    end
end

#########################################################################
# Dictionary conversions (copies)

function PyObject(d::Associative)
    o = PyObject(@pycheckn ccall((@pysym :PyDict_New), PyPtr, ()))
    for k in keys(d)
        @pycheckz ccall((@pysym :PyDict_SetItem), Cint, (PyPtr,PyPtr,PyPtr),
                         o, PyObject(k), PyObject(d[k]))
    end
    return o
end

function convert{K,V}(::Type{Dict{K,V}}, o::PyObject)
    copy(PyDict{K,V}(o))
end

#########################################################################
# Range: integer ranges are converted to xrange,
#         while other ranges (<: AbstractVector) are converted to lists

xrange(start, stop, step) = pycall(pyxrange, PyObject,
                                   start, stop, step)

function PyObject{T<:Integer}(r::Range{T})
    s = step(r)
    f = first(r)
    l = last(r) + s
    if max(f,l) > typemax(Clong) || min(f,l) < typemin(Clong)
        # in Python 2.x, xrange is limited to Clong
        PyObject(T[r...])
    else
        xrange(f, l, s)
    end
end

function convert{T<:Range}(::Type{T}, o::PyObject)
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
# BigFloat and Complex{BigFloat}: convert to/from Python mpmath types

# load mpmath module & initialize.  Currently, this is done
# the first time a BigFloat is converted to Python.  Alternatively,
# we could do it when PyCall is initialized (if mpmath is available),
# at the cost of slowing down initialization in the common case where
# BigFloat conversion is not needed.
const mpprec = [0]
const mpmath = PyNULL()
const mpf = PyNULL()
const mpc = PyNULL()
function mpmath_init()
    if mpmath.o == C_NULL
        copy!(mpmath, pyimport("mpmath"))
        copy!(mpf, mpmath["mpf"])
        copy!(mpc, mpmath["mpc"])
    end
    curprec = get_bigfloat_precision()
    if mpprec[1] != curprec
        mpprec[1] = curprec
        mpmath["mp"]["prec"] = mpprec[1]
    end
end

# TODO: When mpmath uses MPFR internally, can we avoid the string conversions?
# Using strings will work regardless of the mpmath backend, but is annoying
# both from a performance perspective and because it is a lossy conversion
# (since strings use a decimal representation, while MPFR is binary).

function PyObject(x::BigFloat)
    mpmath_init()
    pycall(mpf, PyObject, string(x))
end

function PyObject(x::Complex{BigFloat})
    mpmath_init()
    pycall(mpc, PyObject, string(real(x)), string(imag(x)))
end

function convert(::Type{BigFloat}, o::PyObject)
    parse(BigFloat,
          convert(AbstractString, PyObject(ccall((@pysym :PyObject_Str),
                                                 PyPtr, (PyPtr,), o))))
end

function convert(::Type{Complex{BigFloat}}, o::PyObject)
    try
        Complex{BigFloat}(convert(BigFloat, o["real"]),
                          convert(BigFloat, o["imag"]))
    catch
        convert(Complex{BigFloat}, convert(Complex{Float64}, o))
    end
end

pymp_query(o::PyObject) = pyisinstance(o, mpf) ? BigFloat : pyisinstance(o, mpc) ? Complex{BigFloat} : Union{}

#########################################################################
# BigInt conversion to Python "long" integers

function PyObject(i::BigInt)
    PyObject(@pycheckn ccall((@pysym :PyLong_FromString), PyPtr,
                             (Ptr{UInt8}, Ptr{Void}, Cint),
                             bytestring(string(i)), C_NULL, 10))
end

function convert(::Type{BigInt}, o::PyObject)
    parse(BigInt, convert(AbstractString, PyObject(ccall((@pysym :PyObject_Str),
                                          PyPtr, (PyPtr,), o))))
end

#########################################################################
# Dates (Calendar time)

include("pydates.jl")
#init_datetime() = nothing
#pydate_query(o) = Union{}

#########################################################################
# Inferring Julia types at runtime from Python objects:
#
# [Note that we sometimes use the PyFoo_Check API and sometimes we use
#  PyObject_IsInstance(o, PyFoo_Type), since sometimes the former API
#  is a macro (hence inaccessible in Julia).]

# A type-query function f(o::PyObject) returns the Julia type
# for use with the convert function, or Union{} if there isn't one.

# TODO: In Python 3.x, the BigInt check here won't work since int == long.
pyint_query(o::PyObject) = pyisinstance(o, @pyglobalobj PyInt_Type) ?
  (pyisinstance(o, @pyglobalobj :PyBool_Type) ? Bool : Int) :
  pyisinstance(o, @pyglobalobj :PyLong_Type) ? BigInt :
  pyisinstance(o, npy_integer) ? Int : Union{}

pyfloat_query(o::PyObject) = pyisinstance(o, @pyglobalobj :PyFloat_Type) ||  pyisinstance(o, npy_floating) ? Float64 : Union{}

pycomplex_query(o::PyObject) =
    pyisinstance(o, @pyglobalobj :PyComplex_Type) ||  pyisinstance(o, npy_complexfloating) ? Complex128 : Union{}

pystring_query(o::PyObject) = pyisinstance(o, @pyglobalobj PyString_Type) ? AbstractString : pyisinstance(o, @pyglobalobj :PyUnicode_Type) ? UTF8String : Union{}

# Given call overloading, all PyObjects are callable already, so
# we never automatically convert to Function.
pyfunction_query(o::PyObject) = Union{}

pynothing_query(o::PyObject) = o.o == pynothing ? Void : Union{}

# we check for "items" attr since PyMapping_Check doesn't do this (it only
# checks for __getitem__) and PyMapping_Check returns true for some
# scipy scalar array members, grrr.
pydict_query(o::PyObject) = pyisinstance(o, @pyglobalobj :PyDict_Type) || (pyquery((@pyglobal :PyMapping_Check), o) && ccall((@pysym :PyObject_HasAttrString), Cint, (PyPtr,Array{UInt8}), o, "items") == 1) ? Dict{PyAny,PyAny} : Union{}

typetuple(Ts) = Tuple{Ts...}

function pysequence_query(o::PyObject)
    # pyquery(:PySequence_Check, o) always succeeds according to the docs,
    # but it seems we need to be careful; I've noticed that things like
    # scipy define "fake" sequence types with intmax lengths and other
    # problems
    if pyisinstance(o, @pyglobalobj :PyTuple_Type)
        len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        return typetuple([pytype_query(PyObject(ccall((@pysym :PySequence_GetItem), PyPtr, (PyPtr,Int), o,i-1)), PyAny) for i = 1:len])
    elseif pyisinstance(o, pyxrange)
        return Range
    elseif ispybytearray(o)
        return Vector{UInt8}
    else
        try
            otypestr = get(o["__array_interface__"], PyObject, "typestr")
            typestr = convert(AbstractString, otypestr)
            T = npy_typestrs[typestr[2:end]]
            if T == PyPtr
                T = PyObject
            end
            return Array{T}
        catch
            # only handle PyList for now
            return pyisinstance(o, @pyglobalobj :PyList_Type) ? Array : Union{}
        end
    end
end

macro return_not_None(ex)
    quote
        T = $ex
        if T != Union{}
            return T
        end
    end
end

let
pytype_queries = Array(Tuple{PyObject,Type},0)
global pytype_mapping, pytype_query
function pytype_mapping(py::PyObject, jl::Type)
    for (i,(p,j)) in enumerate(pytype_queries)
        if p == py
            pytype_queries[i] = (py,jl)
            return pytype_queries
        end
    end
    push!(pytype_queries, (py,jl))
end
function pytype_query(o::PyObject, default::Type)
    # TODO: Use some kind of hashtable (e.g. based on PyObject_Type(o)).
    #       (A bit tricky to correctly handle Tuple and other containers.)
    @return_not_None pyint_query(o)
    @return_not_None pyfloat_query(o)
    @return_not_None pycomplex_query(o)
    @return_not_None pystring_query(o)
    @return_not_None pyfunction_query(o)
    @return_not_None pydate_query(o)
    @return_not_None pydict_query(o)
    @return_not_None pysequence_query(o)
    @return_not_None pyptr_query(o)
    @return_not_None pynothing_query(o)
    @return_not_None pymp_query(o)
    for (py,jl) in pytype_queries
        if pyisinstance(o, py)
            return jl
        end
    end
    return default
end
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
        convert(T, o)
    catch
        pyerr_clear() # just in case
        o
    end
end

#########################################################################
# Iteration

function start(po::PyObject)
    sigatomic_begin()
    try
        o = PyObject(@pycheckn ccall((@pysym :PyObject_GetIter), PyPtr, (PyPtr,), po))
        nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), o))

        return (nxt,o)
    finally
        sigatomic_end()
    end
end

function next(po::PyObject, s)
    sigatomic_begin()
    try
        nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), s[2]))
        return (convert(PyAny, s[1]), (nxt, s[2]))
    finally
        sigatomic_end()
    end
end

done(po::PyObject, s) = s[1].o == C_NULL

# issue #216
function Base.collect{T}(::Type{T}, o::PyObject)
    a = Array(T, 0)
    for x in o
        push!(a, x)
    end
    return a
end
Base.collect(o::PyObject) = collect(Any, o)

#########################################################################
