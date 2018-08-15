# Conversions between Julia and Python types for the PyCall module.

#########################################################################
# Conversions of simple types (numbers and nothing)

# conversions from Julia types to PyObject:

@static if pyversion < v"3"
    PyObject(i::Unsigned) = PyObject(@pycheckn ccall(@pysym(:PyInt_FromSize_t),
                                                    PyPtr, (UInt,), i))
    PyObject(i::Integer) = PyObject(@pycheckn ccall(@pysym(:PyInt_FromSsize_t),
                                                    PyPtr, (Int,), i))
else
    PyObject(i::Unsigned) = PyObject(@pycheckn ccall(@pysym(:PyLong_FromUnsignedLongLong),
                                                    PyPtr, (Culonglong,), i))
    PyObject(i::Integer) = PyObject(@pycheckn ccall(@pysym(:PyLong_FromLongLong),
                                                    PyPtr, (Clonglong,), i))
end

PyObject(b::Bool) = PyObject(@pycheckn ccall((@pysym :PyBool_FromLong),
                                             PyPtr, (Clong,), b))

PyObject(r::Real) = PyObject(@pycheckn ccall((@pysym :PyFloat_FromDouble),
                                             PyPtr, (Cdouble,), r))

PyObject(c::Complex) = PyObject(@pycheckn ccall((@pysym :PyComplex_FromDoubles),
                                                PyPtr, (Cdouble,Cdouble),
                                                real(c), imag(c)))

PyObject(n::Nothing) = pyerr_check("PyObject(nothing)", pyincref(pynothing[]))

# conversions to Julia types from PyObject

@static if pyversion < v"3"
    convert(::Type{T}, po::PyObject) where {T<:Integer} =
        T(@pycheck ccall(@pysym(:PyInt_AsSsize_t), Int, (PyPtr,), po))
elseif pyversion < v"3.2"
    convert(::Type{T}, po::PyObject) where {T<:Integer} =
        T(@pycheck ccall(@pysym(:PyLong_AsLongLong), Clonglong, (PyPtr,), po))
else
    function convert(::Type{T}, po::PyObject) where {T<:Integer}
        overflow = Ref{Cint}()
        val = T(@pycheck ccall(@pysym(:PyLong_AsLongLongAndOverflow), Clonglong, (PyPtr, Ref{Cint}), po, overflow))
        iszero(overflow[]) || throw(InexactError(:convert, T, po))
        return val
    end
    function convert(::Type{Integer}, po::PyObject)
        overflow = Ref{Cint}()
        val = @pycheck ccall(@pysym(:PyLong_AsLongLongAndOverflow), Clonglong, (PyPtr, Ref{Cint}), po, overflow)
        iszero(overflow[]) || return convert(BigInt, po)
        return val
    end
end

convert(::Type{Bool}, po::PyObject) =
    0 != @pycheck ccall(@pysym(:PyObject_IsTrue), Cint, (PyPtr,), po)

convert(::Type{T}, po::PyObject) where {T<:Real} =
    T(@pycheck ccall(@pysym(:PyFloat_AsDouble), Cdouble, (PyPtr,), po))

convert(::Type{T}, po::PyObject) where T<:Complex =
    T(@pycheck ccall(@pysym(:PyComplex_AsCComplex), Complex{Cdouble}, (PyPtr,), po))

convert(::Type{Nothing}, po::PyObject) = nothing

#########################################################################
# String conversions (both bytes arrays and unicode strings)

function PyObject(s::AbstractString)
    sb = String(s)
    if pyunicode_literals || !isascii(sb)
        PyObject(@pycheckn ccall(@pysym(PyUnicode_DecodeUTF8),
                                 PyPtr, (Ptr{UInt8}, Int, Ptr{UInt8}),
                                 sb, sizeof(sb), C_NULL))
    else
        pybytes(sb)
    end
end

const _ps_ptr= Ptr{UInt8}[C_NULL]
const _ps_len = Int[0]
function convert(::Type{T}, po::PyObject) where T<:AbstractString
    if pyisinstance(po, @pyglobalobj :PyUnicode_Type)
        convert(T, PyObject(@pycheckn ccall(@pysym(PyUnicode_AsUTF8String),
                                             PyPtr, (PyPtr,), po)))
    else
        @pycheckz ccall(@pysym(PyString_AsStringAndSize),
                        Cint, (PyPtr, Ptr{Ptr{UInt8}}, Ptr{Int}),
                        po, _ps_ptr, _ps_len)
        convert(T, unsafe_string(_ps_ptr[1], _ps_len[1]))
    end
end

# TODO: should symbols be converted to a subclass of Python strings/bytes,
#       so that PyAny conversion can convert it back to a Julia symbol?
PyObject(s::Symbol) = PyObject(string(s))
convert(::Type{Symbol}, po::PyObject) = Symbol(convert(AbstractString, po))

#########################################################################
# ByteArray conversions

function PyObject(a::DenseVector{UInt8})
  if stride(a,1) != 1
    try
        return NpyArray(a, true)
    catch
        return array2py(a) # fallback to non-NumPy version
    end
  end
  PyObject(@pycheckn ccall((@pysym :PyByteArray_FromStringAndSize),
                           PyPtr, (Ptr{UInt8}, Int), a, length(a)))
end

ispybytearray(po::PyObject) =
  pyisinstance(po, @pyglobalobj :PyByteArray_Type)

function convert(::Type{Vector{UInt8}}, po::PyObject)
    b = PyBuffer(po)
    iscontiguous(b) || error("a contiguous buffer is required")
    return copy(unsafe_wrap(Array, Ptr{UInt8}(pointer(b)), sizeof(b)))
end

# TODO: support zero-copy PyByteArray <: AbstractVector{UInt8} object

#########################################################################
# Pointer conversions, using ctypes or PyCapsule

PyObject(p::Ptr) = pycall(c_void_p_Type, PyObject, UInt(p))

function convert(::Type{Ptr{Cvoid}}, po::PyObject)
    if pyisinstance(po, c_void_p_Type)
        v = po["value"]
        # ctypes stores the NULL pointer specially, grrr
        pynothing_query(v) == Nothing ? C_NULL :
          convert(Ptr{Cvoid}, convert(UInt, po["value"]))
    elseif pyisinstance(po, @pyglobalobj(:PyCapsule_Type))
        @pycheck ccall((@pysym :PyCapsule_GetPointer),
                       Ptr{Cvoid}, (PyPtr,Ptr{UInt8}),
                       po, ccall((@pysym :PyCapsule_GetName),
                                 Ptr{UInt8}, (PyPtr,), po))
    else
        convert(Ptr{Cvoid}, convert(UInt, po))
    end
end

pyptr_query(po::PyObject) = pyisinstance(po, c_void_p_Type) || pyisinstance(po, @pyglobalobj(:PyCapsule_Type)) ? Ptr{Cvoid} : Union{}

#########################################################################
# for automatic conversions, I pass Vector{PyAny}, NTuple{N, PyAny}, etc.,
# but since PyAny is an abstract type I need to convert this to Any
# before actually creating the Julia object

# I want to use a union, but this seems to confuse Julia's method
# dispatch for the convert function in some circumstances
# const PyAny = Union{PyObject, Int, Bool, Float64, ComplexF64, AbstractString, Function, Dict, Tuple, Array}
abstract type PyAny end

function pyany_toany(T::Type)
    T === Vararg{PyAny} ? Vararg{Any} : T
end
pyany_toany(::Type{PyAny}) = Any
pyany_toany(t::Type{T}) where {T<:Tuple} = Tuple{map(pyany_toany, t.types)...}

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
    len = lastindex(t) # lastindex, not length, because of julia#14924
    o = PyObject(@pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), len))
    for i = 1:len
        oi = PyObject(t[i])
        @pycheckz ccall((@pysym :PyTuple_SetItem), Cint, (PyPtr,Int,PyPtr),
                         o, i-1, oi)
        pyincref(oi) # PyTuple_SetItem steals the reference
    end
    return o
end

# somewhat annoying to get the length and types in a tuple type
# ... would be better not to have to use undocumented internals!
istuplen(T,isva,n) = isva ? n ≥ length(T.parameters)-1 : n == length(T.parameters)
function tuptype(T::DataType,isva,i)
    if isva && i ≥ length(T.parameters)
        return Base.unwrapva(T.parameters[end])
    else
        return T.parameters[i]
    end
end
tuptype(T::UnionAll,isva,i) = tuptype(T.body,isva,i)
isvatuple(T::UnionAll) = isvatuple(T.body)
isvatuple(T::DataType) = !isempty(T.parameters) && Base.isvarargtype(T.parameters[end])

function convert(tt::Type{T}, o::PyObject) where T<:Tuple
    isva = isvatuple(T)
    len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
    if !istuplen(tt, isva, len)
        throw(BoundsError())
    end
    ntuple((i ->
            convert(tuptype(T, isva, i),
                    PyObject(ccall((@pysym :PySequence_GetItem), PyPtr,
                                   (PyPtr, Int), o, i-1)))),
           len)
end

function convert(::Type{Pair{K,V}}, o::PyObject) where {K,V}
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
mutable struct PyVector{T} <: AbstractVector{T}
    o::PyObject
    function PyVector{T}(o::PyObject) where T
        if ispynull(o)
            throw(ArgumentError("cannot make PyVector from NULL PyObject"))
        end
        new{T}(o)
    end
end

PyVector(o::PyObject) = PyVector{PyAny}(o)
PyObject(a::PyVector) = a.o
convert(::Type{PyVector}, o::PyObject) = PyVector(o)
convert(::Type{PyVector{T}}, o::PyObject) where {T} = PyVector{T}(o)
unsafe_convert(::Type{PyPtr}, a::PyVector) = a.o.o
PyVector(a::PyVector) = a
PyVector(a::AbstractVector{T}) where {T} = PyVector{T}(array2py(a))

# when a PyVector is copied it is converted into an ordinary Julia Vector
similar(a::PyVector, T, dims::Dims) = Array{T}(dims)
similar(a::PyVector{T}) where {T} = similar(a, pyany_toany(T), size(a))
similar(a::PyVector{T}, dims::Dims) where {T} = similar(a, pyany_toany(T), dims)
similar(a::PyVector{T}, dims::Int...) where {T} = similar(a, pyany_toany(T), dims)
eltype(::PyVector{T}) where {T} = pyany_toany(T)
eltype(::Type{PyVector{T}}) where {T} = pyany_toany(T)

size(a::PyVector) = (length(a.o),)

getindex(a::PyVector) = getindex(a, 1)
getindex(a::PyVector{T}, i::Integer) where {T} = convert(T, PyObject(@pycheckn ccall((@pysym :PySequence_GetItem), PyPtr, (PyPtr, Int), a, i-1)))

setindex!(a::PyVector, v) = setindex!(a, v, 1)
function setindex!(a::PyVector, v, i::Integer)
    @pycheckz ccall((@pysym :PySequence_SetItem), Cint, (PyPtr, Int, PyPtr), a, i-1, PyObject(v))
    v
end

summary(a::PyVector{T}) where {T} = string(Base.dims2string(size(a)), " ",
                                          string(pyany_toany(T)), " PyVector")

splice!(a::PyVector, i::Integer) = splice!(a.o, i)
function splice!(a::PyVector{T}, indices::AbstractVector{I}) where {T,I<:Integer}
    v = pyany_toany(T)[a[i] for i in indices]
    for i in sort(indices, rev=true)
        @pycheckz ccall((@pysym :PySequence_DelItem), Cint, (PyPtr, Int), a, i-1)
    end
    v
end
pop!(a::PyVector) = pop!(a.o)
popfirst!(a::PyVector) = popfirst!(a.o)
empty!(a::PyVector) = empty!(a.o)

# only works for List subtypes:
push!(a::PyVector, item) = push!(a.o, item)
insert!(a::PyVector, i::Integer, item) = insert!(a.o, i, item)
pushfirst!(a::PyVector, item) = pushfirst!(a.o, item)
prepend!(a::PyVector, items) = prepend!(a.o, items)
append!(a::PyVector{T}, items) where {T} = PyVector{T}(append!(a.o, items))

#########################################################################
# Lists and 1d arrays.

# recursive conversion of A to a list of list of lists... starting
# with dimension dim and index i in A.
function array2py(A::AbstractArray{T, N}, dim::Integer, i::Integer) where {T, N}
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
   ndims(A) <= 1 || hasmethod(stride, Tuple{typeof(A),Int}) ? array2py(A) :
   pyjlwrap_new(A)

function py2array(T, A::Array{TA,N}, o::PyObject,
                  dim::Integer, i::Integer) where {TA,N}
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
    A = Array{pyany_toany(T)}(undef, dims)
    py2array(T, A, o, 1, 1)
end

function convert(::Type{Vector{T}}, o::PyObject) where T
    len = ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
    if len < 0 || # not a sequence
       len+1 < 0  # object pretending to be a sequence of infinite length
        pyerr_clear()
        throw(ArgumentError("expected Python sequence"))
    end
    py2array(T, Array{pyany_toany(T)}(undef, len), o, 1, 1)
end

convert(::Type{Array}, o::PyObject) = map(identity, py2array(PyAny, o))
convert(::Type{Array{T}}, o::PyObject) where {T} = py2array(T, o)

PyObject(a::BitArray) = PyObject(Array(a))

# NumPy conversions (multidimensional arrays)
include("numpy.jl")

#########################################################################
# PyDict: no-copy wrapping of a Julia object around a Python dictionary

# we check for "items" attr since PyMapping_Check doesn't do this (it only
# checks for __getitem__) and PyMapping_Check returns true for some
# scipy scalar array members, grrr.
function is_mapping_object(o::PyObject)
    pyisinstance(o, @pyglobalobj :PyDict_Type) ||
    (pyquery((@pyglobal :PyMapping_Check), o) &&
      ccall((@pysym :PyObject_HasAttrString), Cint, (PyPtr,Ptr{UInt8}), o, "items") == 1)
end

"""
    PyDict(o::PyObject)
    PyDict(d::Dict{K,V})

This returns a PyDict, which is a no-copy wrapper around a Python dictionary.

Alternatively, you can specify the return type of a `pycall` as PyDict.
"""
mutable struct PyDict{K,V,isdict} <: AbstractDict{K,V}
    o::PyObject
    # isdict = true for python dict, otherwise is a generic Mapping object

    function PyDict{K,V,isdict}(o::PyObject) where {K,V,isdict}
        if !isdict && !ispynull(o) && !is_mapping_object(o)
            throw(ArgumentError("only Dict and Mapping objects can be converted to PyDict"))
        end
        return new{K,V,isdict}(o)
    end
end

PyDict{K,V}(o::PyObject) where {K,V} = PyDict{K,V,pyisinstance(o, @pyglobalobj :PyDict_Type)}(o)
PyDict{K,V}() where {K,V} = PyDict{K,V,true}(PyObject(@pycheckn ccall((@pysym :PyDict_New), PyPtr, ())))

PyDict(o::PyObject) = PyDict{PyAny,PyAny}(o)
PyObject(d::PyDict) = d.o
PyDict() = PyDict{PyAny,PyAny}()
PyDict(d::AbstractDict{K,V}) where {K,V} = PyDict{K,V}(PyObject(d))
PyDict(d::AbstractDict{Any,Any}) = PyDict{PyAny,PyAny}(PyObject(d))
PyDict(d::AbstractDict{Any,V}) where {V} = PyDict{PyAny,V}(PyObject(d))
PyDict(d::AbstractDict{K,Any}) where {K} = PyDict{K,PyAny}(PyObject(d))
convert(::Type{PyDict}, o::PyObject) = PyDict(o)
convert(::Type{PyDict{K,V}}, o::PyObject) where {K,V} = PyDict{K,V}(o)
unsafe_convert(::Type{PyPtr}, d::PyDict) = d.o.o

haskey(d::PyDict{K,V,true}, key) where {K,V} = 1 == ccall(@pysym(:PyDict_Contains), Cint, (PyPtr, PyPtr), d, PyObject(key))
keys(::Type{T}, d::PyDict{K,V,true}) where {T,K,V} = convert(Vector{T}, PyObject(@pycheckn ccall((@pysym :PyDict_Keys), PyPtr, (PyPtr,), d)))
values(::Type{T}, d::PyDict{K,V,true}) where {T,K,V} = convert(Vector{T}, PyObject(@pycheckn ccall((@pysym :PyDict_Values), PyPtr, (PyPtr,), d)))

keys(::Type{T}, d::PyDict{K,V,false}) where {T,K,V} = convert(Vector{T}, pycall(d.o["keys"], PyObject))
values(::Type{T}, d::PyDict{K,V,false}) where {T,K,V} = convert(Vector{T}, pycall(d.o["values"], PyObject))
haskey(d::PyDict{K,V,false}, key) where {K,V} = 1 == ccall(@pysym(:PyMapping_HasKey), Cint, (PyPtr, PyPtr), d, PyObject(key))

similar(d::PyDict{K,V}) where {K,V} = Dict{pyany_toany(K),pyany_toany(V)}()
eltype(::Type{PyDict{K,V}}) where {K,V} = Pair{pyany_toany(K),pyany_toany(V)}
Base.keytype(::PyDict{K,V}) where {K,V} = pyany_toany(K)
Base.valtype(::PyDict{K,V}) where {K,V} = pyany_toany(V)
Base.keytype(::Type{PyDict{K,V}}) where {K,V} = pyany_toany(K)
Base.valtype(::Type{PyDict{K,V}}) where {K,V} = pyany_toany(V)

function setindex!(d::PyDict, v, k)
    @pycheckz ccall((@pysym :PyObject_SetItem), Cint, (PyPtr, PyPtr, PyPtr),
                     d, PyObject(k), PyObject(v))
    v
end

get(d::PyDict{K,V}, k, default) where {K,V} = get(d.o, V, k, default)

function pop!(d::PyDict{K,V,true}, k) where {K,V}
    v = d[k]
    @pycheckz ccall(@pysym(:PyDict_DelItem), Cint, (PyPtr, PyPtr), d, PyObject(k))
    return v
end
function pop!(d::PyDict{K,V,false}, k) where {K,V}
    v = d[k]
    @pycheckz ccall(@pysym(:PyObject_DelItem), Cint, (PyPtr, PyPtr), d, PyObject(k))
    return v
end

function pop!(d::PyDict, k, default)
    try
        return pop!(d, k)
    catch
        return default
    end
end

function delete!(d::PyDict{K,V,true}, k) where {K,V}
    e = ccall(@pysym(:PyDict_DelItem), Cint, (PyPtr, PyPtr), d, PyObject(k))
    e == -1 && pyerr_clear() # delete! ignores errors in Julia
    return d
end
function delete!(d::PyDict{K,V,false}, k) where {K,V}
    e = ccall(@pysym(:PyObject_DelItem), Cint, (PyPtr, PyPtr), d, PyObject(k))
    e == -1 && pyerr_clear() # delete! ignores errors in Julia
    return d
end

function empty!(d::PyDict{K,V,true}) where {K,V}
    @pycheck ccall((@pysym :PyDict_Clear), Cvoid, (PyPtr,), d)
    return d
end
function empty!(d::PyDict{K,V,false}) where {K,V}
    # for generic Mapping items we must delete keys one by one
    for k in keys(d)
        delete!(d, k)
    end
    return d
end

length(d::PyDict{K,V,true}) where {K,V} = @pycheckz ccall(@pysym(:PyDict_Size), Int, (PyPtr,), d)
length(d::PyDict{K,V,false}) where {K,V} = @pycheckz ccall(@pysym(:PyObject_Size), Int, (PyPtr,), d)
isempty(d::PyDict) = length(d) == 0


struct PyDict_Iterator
    # arrays to pass key, value, and pos pointers to PyDict_Next
    ka::Ref{PyPtr}
    va::Ref{PyPtr}
    pa::Ref{Int}
    i::Int # current position in items list (0-based)
    len::Int # length of items list
end
@static if VERSION < v"0.7.0-DEV.5126" # julia#25261
    Base.start(d::PyDict{K,V,true}) where {K,V} = PyDict_Iterator(Ref{PyPtr}(), Ref{PyPtr}(), Ref(0), 0, length(d))
    Base.done(d::PyDict{K,V,true}, itr::PyDict_Iterator) where {K,V} = itr.i >= itr.len
    function Base.next(d::PyDict{K,V,true}, itr::PyDict_Iterator) where {K,V}
        if 0 == ccall((@pysym :PyDict_Next), Cint,
                      (PyPtr, Ref{Int}, Ref{PyPtr}, Ref{PyPtr}),
                      d, itr.pa, itr.ka, itr.va)
            error("unexpected end of PyDict_Next")
        end
        ko = pyincref(itr.ka[]) # PyDict_Next returns
        vo = pyincref(itr.va[]) #   borrowed ref, so incref
        (Pair(convert(K,ko), convert(V,vo)),
         PyDict_Iterator(itr.ka, itr.va, itr.pa, itr.i+1, itr.len))
     end

    # Iterator for generic mapping, using Python items iterator.
    # To strictly use the Julia iteration protocol, we should pass
    # d.o["items"] rather than d.o to done and next, but the PyObject
    # iterator functions only look at the state s, so we are okay.
    Base.start(d::PyDict{K,V,false}) where {K,V} = start(pycall(d.o["items"], PyObject))
    Base.done(d::PyDict{K,V,false}, s) where {K,V} = done(d.o, s)
    function Base.next(d::PyDict{K,V,false}, s) where {K,V}
        nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), s[2]))
        return (convert(Pair{K,V}, s[1]), (nxt, s[2]))
    end
else
    function Base.iterate(d::PyDict{K,V,true}, itr=PyDict_Iterator(Ref{PyPtr}(), Ref{PyPtr}(), Ref(0), 0, length(d))) where {K,V}
        itr.i >= itr.len && return nothing
        if 0 == ccall((@pysym :PyDict_Next), Cint,
                      (PyPtr, Ref{Int}, Ref{PyPtr}, Ref{PyPtr}),
                      d, itr.pa, itr.ka, itr.va)
            error("unexpected end of PyDict_Next")
        end
        ko = pyincref(itr.ka[]) # PyDict_Next returns
        vo = pyincref(itr.va[]) #   borrowed ref, so incref
        (Pair(convert(K,ko), convert(V,vo)),
         PyDict_Iterator(itr.ka, itr.va, itr.pa, itr.i+1, itr.len))
    end

    # Iterator for generic mapping, using Python items iterator.
    # Our approach is to wrap an iterator over d.o["items"]
    # which necessitates including d.o["items"] in the state.
    function _start(d::PyDict{K,V,false}) where {K,V}
        d_items = pycall(d.o["items"], PyObject)
        (d_items, iterate(d_items))
    end
    function Base.iterate(d::PyDict{K,V,false}, itr=_start(d)) where {K,V}
        d_items, iter_result = itr
        iter_result === nothing && return nothing
        item, state = iter_result
        iter_result = iterate(d_items, state)
        (item[1] => item[2], (d_items, iter_result))
    end
end

#########################################################################
# Dictionary conversions (copies)

function PyObject(d::AbstractDict)
    o = PyObject(@pycheckn ccall((@pysym :PyDict_New), PyPtr, ()))
    for k in keys(d)
        @pycheckz ccall((@pysym :PyDict_SetItem), Cint, (PyPtr,PyPtr,PyPtr),
                         o, PyObject(k), PyObject(d[k]))
    end
    return o
end

function convert(::Type{Dict{K,V}}, o::PyObject) where {K,V}
    copy(PyDict{K,V}(o))
end

#########################################################################
# AbstractRange: integer ranges are converted to xrange,
#                while other ranges (<: AbstractVector) are converted to lists

xrange(start, stop, step) = pycall(pyxrange[], PyObject,
                                   start, stop, step)

function PyObject(r::AbstractRange{T}) where T<:Integer
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

function convert(::Type{T}, o::PyObject) where T<:AbstractRange
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
    if ispynull(mpmath)
        copy!(mpmath, pyimport("mpmath"))
        copy!(mpf, mpmath["mpf"])
        copy!(mpc, mpmath["mpc"])
    end
    curprec = precision(BigFloat)
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

convert(::Type{BigFloat}, o::PyObject) = parse(BigFloat, pystr(o))

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
# (Int64), Int128 and BigInt conversion to Python "long" integers

const LongInt = @static (Sys.WORD_SIZE==32) ? Union{Int64,UInt64,Int128,UInt128,BigInt} : Union{Int128,UInt128,BigInt}

function PyObject(i::LongInt)
    PyObject(@pycheckn ccall((@pysym :PyLong_FromString), PyPtr,
                             (Ptr{UInt8}, Ptr{Cvoid}, Cint),
                             String(string(i)), C_NULL, 10))
end

convert(::Type{BigInt}, o::PyObject) = parse(BigInt, pystr(o))

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

@static if pyversion < v"3"
    pyint_query(o::PyObject) = pyisinstance(o, @pyglobalobj :PyInt_Type) ?
        (pyisinstance(o, @pyglobalobj :PyBool_Type) ? Bool : Int) :
        pyisinstance(o, @pyglobalobj :PyLong_Type) ? BigInt :
        pyisinstance(o, npy_integer) ? Int : Union{}
else
    pyint_query(o::PyObject) = pyisinstance(o, @pyglobalobj :PyLong_Type) ?
        (pyisinstance(o, @pyglobalobj :PyBool_Type) ? Bool : Integer) :
        pyisinstance(o, npy_integer) ? Integer : Union{}
end

pyfloat_query(o::PyObject) = pyisinstance(o, @pyglobalobj :PyFloat_Type) ||  pyisinstance(o, npy_floating) ? Float64 : Union{}

pycomplex_query(o::PyObject) =
    pyisinstance(o, @pyglobalobj :PyComplex_Type) ||  pyisinstance(o, npy_complexfloating) ? ComplexF64 : Union{}

pystring_query(o::PyObject) = pyisinstance(o, @pyglobalobj PyString_Type) ? AbstractString : pyisinstance(o, @pyglobalobj :PyUnicode_Type) ? String : Union{}

# Given call overloading, all PyObjects are callable already, so
# we never automatically convert to Function.
pyfunction_query(o::PyObject) = Union{}

pynothing_query(o::PyObject) = o.o == pynothing[] ? Nothing : Union{}

# We refrain from converting all objects that support the mapping protocol (PyMapping_Check)
# to avoid converting types like Pandas `DataFrame` that are only lossily
# representable as a Julia dictionary (issue #376).
pydict_query(o::PyObject) = pyisinstance(o, @pyglobalobj :PyDict_Type) ? Dict{PyAny,PyAny} : Union{}

typetuple(Ts) = Tuple{Ts...}

function pysequence_query(o::PyObject)
    # pyquery(:PySequence_Check, o) always succeeds according to the docs,
    # but it seems we need to be careful; I've noticed that things like
    # scipy define "fake" sequence types with intmax lengths and other
    # problems
    if pyisinstance(o, @pyglobalobj :PyTuple_Type)
        len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        return typetuple(pytype_query(PyObject(ccall((@pysym :PySequence_GetItem), PyPtr, (PyPtr,Int), o,i-1)), PyAny) for i = 1:len)
    elseif pyisinstance(o, pyxrange[])
        return AbstractRange
    elseif ispybytearray(o)
        return Vector{UInt8}
    elseif !haskey(o, "__array_interface__")
        # only handle PyList for now
        return pyisinstance(o, @pyglobalobj :PyList_Type) ? Array : Union{}
    else
        otypestr = get(o["__array_interface__"], PyObject, "typestr")
        typestr = convert(AbstractString, otypestr) # Could this just be String now?
        T = npy_typestrs[typestr[2:end]]
        if T == PyPtr
            T = PyObject
        end
        return Array{T}
    end
end

macro return_not_None(ex)
    quote
        T = $(esc(ex))
        if T != Union{}
            return T
        end
    end
end

const pytype_queries = Tuple{PyObject,Type}[]
"""
    pytype_mapping(pytype, jltype)

Given a Python type object `pytype`, tell PyCall to convert it to
`jltype` in `PyAny(object)` conversions.
"""
function pytype_mapping(py::PyObject, jl::Type)
    for (i,(p,j)) in enumerate(pytype_queries)
        if p == py
            pytype_queries[i] = (py,jl)
            return pytype_queries
        end
    end
    push!(pytype_queries, (py,jl))
end
"""
    pytype_query(o::PyObject, default=PyObject)

Given a Python object `o`, return the corresponding
native Julia type (defaulting to `default`) that we convert
`o` to in `PyAny(o)` conversions.
"""
function pytype_query(o::PyObject, default::TypeTuple=PyObject)
    # TODO: Use some kind of hashtable (e.g. based on PyObject_Type(o)).
    #       (A bit tricky to correctly handle Tuple and other containers.)
    @return_not_None pyint_query(o)
    pyisinstance(o, npy_bool) && return Bool
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

function convert(::Type{PyAny}, o::PyObject)
    if ispynull(o)
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
