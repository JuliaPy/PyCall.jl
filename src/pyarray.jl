#########################################################################
# Extract shape and other information about arrays that support Python's
# Buffer Interface/Protocol (PEP 3118)
#########################################################################
struct PyArray_Info{T,N}
    native::Bool # native byte order?
    sz::NTuple{N,Int}
    st::NTuple{N,Int} # strides, in multiples of bytes!
    data::Ptr{T}
    readonly::Bool
end

function PyArray_Info(o::PyObject)
    # n.b. the pydecref(::PyBuffer) finalizer handles releasing the PyBuffer
    pybuf = PyBuffer(o, PyBUF_ND_CONTIGUOUS)
    T, native_byteorder = array_format(pybuf)
    sz = size(pybuf)
    strd = strides(pybuf)
    length(strd) == 0 && (sz = ())
    N = length(sz)
    isreadonly = pybuf.buf.readonly==1
    return PyArray_Info{T,N}(native_byteorder, sz, strd, pybuf.buf.buf, isreadonly)
end

aligned(i::PyArray_Info{T,N}) where {T,N} = #  FIXME: also check pointer alignment?
  all(m -> m == 0, mod.(i.st, sizeof(T))) # strides divisible by elsize

eltype(i::PyArray_Info{T,N}) where {T,N} = T
ndims(i::PyArray_Info{T,N})  where {T,N} = N

function default_stride(sz::NTuple{N, Int}, ::Type{T}) where {T,N}
    stv = Vector{Int}(N)
    stv[end] = sizeof(T)
    for i = N-1:-1:1
        stv[i] = stv[i+1]*sz[i+1]
    end
    ntuple(i->stv[i], N)
end

# whether a contiguous array in column-major (Fortran, Julia) order
function f_contiguous(T::Type, sz::NTuple{N,Int}, st::NTuple{N,Int}) where N
    if prod(sz) == 1 || length(sz) == 1
        # 0 or 1-dim arrays should default to f-contiguous in julia
        return true
    end
    if st[1] != sizeof(T)
        return false
    end
    for j = 2:N
        if st[j] != st[j-1] * sz[j-1]
            return false
        end
    end
    return true
end

f_contiguous(T::Type, sz::NTuple{N1,Int}, st::NTuple{N2,Int}) where {N1,N2} =
    error("stride and size are different lengths, size: $sz, strides: $sz")

f_contiguous(i::PyArray_Info{T,N}) where {T,N} = f_contiguous(T, i.sz, i.st)
c_contiguous(i::PyArray_Info{T,N}) where {T,N} = f_contiguous(T, reverse(i.sz), reverse(i.st))

#########################################################################
# PyArray: no-copy wrapper around NumPy ndarray
#
# Hopefully, in the future this can be a subclass of StridedArray (see
# Julia issue #2345), which will allow it to be used with most Julia
# functions, but that is not possible at the moment.  So, to use this
# with Julia linalg functions etcetera a copy is still required.

"""
    PyArray(o::PyObject)

This converts an `ndarray` object `o` to a PyArray.

This implements a nocopy wrapper to a NumPy array (currently of only numeric types only).

If you are using `pycall` and the function returns an `ndarray`, you can use `PyArray` as the return type to directly receive a `PyArray`.
"""
mutable struct PyArray{T,N} <: AbstractArray{T,N}
    o::PyObject
    info::PyArray_Info
    dims::Dims
    st::NTuple{N,Int}
    f_contig::Bool
    c_contig::Bool
    data::Ptr{T}

    function PyArray{T,N}(o::PyObject, info::PyArray_Info) where {T,N}
        if !aligned(info)
            throw(ArgumentError("only NPY_ARRAY_ALIGNED arrays are supported"))
        elseif !info.native
            throw(ArgumentError("only native byte-order arrays are supported"))
        elseif eltype(info) != T
            throw(ArgumentError("inconsistent type in PyArray constructor"))
        elseif length(info.sz) != N || length(info.st) != N
            throw(ArgumentError("inconsistent ndims in PyArray constructor"))
        end
        return new{T,N}(o, info, tuple(info.sz...), div.(info.st, sizeof(T)),
                        f_contiguous(info), c_contiguous(info),
                        convert(Ptr{T}, info.data))
    end
end

function PyArray(o::PyObject)
    info = PyArray_Info(o)
    return PyArray{eltype(info), length(info.sz)}(o, info)
end

size(a::PyArray) = a.dims
ndims(a::PyArray{T,N}) where {T,N} = N

similar(a::PyArray, T, dims::Dims) = Array{T}(undef, dims)

function setdata!{T,N}(a::PyArray{T,N}, o::PyObject, pybufinfo=PyBuffer())
    PyBuffer!(pybufinfo, o, PyBUF_ND_CONTIGUOUS)
    dataptr = pybufinfo.buf.buf
    a.data = reinterpret(Ptr{T}, dataptr)
    a
end

function copy(a::PyArray{T,N}) where {T,N}
    if N > 1 && a.c_contig # equivalent to f_contig with reversed dims
        B = unsafe_wrap(Array, a.data, ntuple((n -> a.dims[N - n + 1]), N))
        return permutedims(B, (N:-1:1))
    end
    A = Array{T}(undef, a.dims)
    if a.f_contig
        ccall(:memcpy, Cvoid, (Ptr{T}, Ptr{T}, Int), A, a, sizeof(T)*length(a))
        return A
    else
        return copyto!(A, a)
    end
end

# TODO: need to do bounds-checking of these indices!

getindex(a::PyArray{T,0}) where {T} = unsafe_load(a.data)
getindex(a::PyArray{T,1}, i::Integer) where {T} = unsafe_load(a.data, 1 + (i-1)*a.st[1])

getindex(a::PyArray{T,2}, i::Integer, j::Integer) where {T} =
  unsafe_load(a.data, 1 + (i-1)*a.st[1] + (j-1)*a.st[2])

function getindex(a::PyArray, i::Integer)
    if a.f_contig
        return unsafe_load(a.data, i)
    else
        return a[ind2sub(a.dims, i)...]
    end
end

function getindex(a::PyArray, is::Integer...)
    index = 1
    n = min(length(is),length(a.st))
    for i = 1:n
        index += (is[i]-1)*a.st[i]
    end
    for i = n+1:length(is)
        if is[i] != 1
            throw(BoundsError())
        end
    end
    unsafe_load(a.data, index)
end

function writeok_assign(a::PyArray, v, i::Integer)
    if a.info.readonly
        throw(ArgumentError("read-only PyArray"))
    else
        unsafe_store!(a.data, v, i)
    end
    return a
end

setindex!(a::PyArray{T,0}, v) where {T} = writeok_assign(a, v, 1)
setindex!(a::PyArray{T,1}, v, i::Integer) where {T} = writeok_assign(a, v, 1 + (i-1)*a.st[1])

setindex!(a::PyArray{T,2}, v, i::Integer, j::Integer) where {T} =
  writeok_assign(a, v, 1 + (i-1)*a.st[1] + (j-1)*a.st[2])

function setindex!(a::PyArray, v, i::Integer)
    if a.f_contig
        return writeok_assign(a, v, i)
    else
        return setindex!(a, v, ind2sub(a.dims, i)...)
    end
end

function setindex!(a::PyArray, v, is::Integer...)
    index = 1
    n = min(length(is),length(a.st))
    for i = 1:n
        index += (is[i]-1)*a.st[i]
    end
    for i = n+1:length(is)
        if is[i] != 1
            throw(BoundsError())
        end
    end
    writeok_assign(a, v, index)
end

stride(a::PyArray, i::Integer) = a.st[i]

Base.unsafe_convert(::Type{Ptr{T}}, a::PyArray{T}) where {T} = a.data

pointer(a::PyArray, i::Int) = pointer(a, ind2sub(a.dims, i))

function pointer(a::PyArray{T}, is::Tuple{Vararg{Int}}) where T
    offset = 0
    for i = 1:length(is)
        offset += (is[i]-1)*a.st[i]
    end
    return a.data + offset*sizeof(T)
end

summary(a::PyArray{T}) where {T} = string(Base.dims2string(size(a)), " ",
                                          string(T), " PyArray")

#########################################################################
# PyArray <-> PyObject conversions

const PYARR_TYPES = Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Float16,Float32,Float64,ComplexF32,ComplexF64,PyPtr}

PyObject(a::PyArray) = a.o

convert(::Type{PyArray}, o::PyObject) = PyArray(o)

function convert(::Type{Array{T, 1}}, o::PyObject) where T<:PYARR_TYPES
    try
        copy(PyArray{T, 1}(o, PyArray_Info(o))) # will check T and N vs. info
    catch
        len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        A = Array{pyany_toany(T)}(undef, len)
        py2array(T, A, o, 1, 1)
    end
end

function convert(::Type{Array{T}}, o::PyObject) where T<:PYARR_TYPES
    try
        info = PyArray_Info(o)
        try
            copy(PyArray{T, length(info.sz)}(o, info)) # will check T == eltype(info)
        catch
            return py2array(T, Array{pyany_toany(T)}(undef, info.sz...), o, 1, 1)
        end
    catch
        py2array(T, o)
    end
end

function convert(::Type{Array{T,N}}, o::PyObject) where {T<:PYARR_TYPES,N}
    try
        info = PyArray_Info(o)
        try
            copy(PyArray{T,N}(o, info)) # will check T,N == eltype(info),ndims(info)
        catch
            nd = length(info.sz)
            if nd != N
                throw(ArgumentError("cannot convert $(nd)d array to $(N)d"))
            end
            return py2array(T, Array{pyany_toany(T)}(undef, info.sz...), o, 1, 1)
        end
    catch
        A = py2array(T, o)
        if ndims(A) != N
            throw(ArgumentError("cannot convert $(ndims(A))d array to $(N)d"))
        end
        A
    end
end

function convert(::Type{Array{PyObject}}, o::PyObject)
    map(pyincref, convert(Array{PyPtr}, o))
end

function convert(::Type{Array{PyObject,1}}, o::PyObject)
    map(pyincref, convert(Array{PyPtr, 1}, o))
end

function convert(::Type{Array{PyObject,N}}, o::PyObject) where N
    map(pyincref, convert(Array{PyPtr, N}, o))
end

array_format(o::PyObject) = array_format(PyBuffer(o, PyBUF_ND_CONTIGUOUS))

"""
```
NoCopyArray(o::PyObject)
```
Convert a Python array-like object, to a Julia `Array` without making a copy of
the data. If the data is stored in row-major format (the default in
Python/NumPy), then the returned array `nca` will be a `PermutedDimsArray` such
that the arrays are indexed the same way in Julia and Python. i.e.
`nca[idxs...] == o[idxs...]`

Warning: This function is only lightly tested, and should be considered
experimental - it may cause segmentation faults on conversion or subsequent
array access, or be subtly broken in other ways. Only dense/contiguous, native
endian arrays that support the Python Buffer protocol are likely be converted
correctly.
"""
function NoCopyArray(o::PyObject)
    # n.b. the pydecref(::PyBuffer) finalizer handles releasing the PyBuffer
    pybuf = PyBuffer(o, PyBUF_ND_CONTIGUOUS)
    T, native_byteorder = array_format(pybuf)
    !native_byteorder && throw(ArgumentError(
      "Only native endian format supported, format string: '$(get_format_str(pybuf))'"))
    T == Void && throw(ArgumentError(
      "Array datatype '$(get_format_str(pybuf))' not supported"))
    # TODO more checks on strides etc
    sz = size(pybuf)
    arr = unsafe_wrap(Array, convert(Ptr{T}, pybuf.buf.buf), sz, false)
    !f_contiguous(T, sz, strides(pybuf)) &&
        (arr = PermutedDimsArray(reshape(arr, reverse(sz)), (pybuf.buf.ndim:-1:1)))
    return arr
end
