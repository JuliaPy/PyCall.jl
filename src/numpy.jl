# Julia wrappers around the NumPy API, as part of the PyCall package

#########################################################################
# Initialization (UGLY)

# Linking NumPy's C API from Julia requires some serious hackery,
# because NumPy does not export its symbols in the usual way for
# shared libraries.  Instead, it provides a Python variable
# numpy.core.multiarray._ARRAY_API that points to a lookup table of
# pointers to the API functions and global variables.  Moreover, the
# meaning of this table, along with a lot of other important
# constants, is defined in a C header file which changes between NumPy
# versions, so we need to do some regex parsing of this header file in
# order to extract the necessary information to call NumPy.  Ugly and
# mildly insane, but I don't see much alternative (at least to do this
# purely in Julia).
#
# The result of npy_api_initialize, below, is to produce the following
# tables of API pointers:
npy_api = Dict{Symbol, Ptr{Void}}()

# type alias to shorten type decorations (needed for JIT) of our table
typealias Tnpy_api Dict{Symbol, Ptr{Void}}

# need a global to cache pyimport("numpy.core.multiarray"), in order
# to ensure the module is not garbage-collected as long as we are using it
# for the npy_api pointers.
npy_multiarray = PyObject()

npy_initialized = false # global to prevent multiple initializations

# Macro version of npyinitialize() to inline npy_initialized? check
macro npyinitialize()
    :(npy_initialized::Bool ? nothing : npyinitialize())
end

function npyinitialize()
    global npy_api
    global npy_initialized
    global npy_multiarray

    if npy_initialized::Bool
        return
    end
    try
        npy_multiarray::PyObject = pyimport("numpy.core.multiarray")
    catch e
        error("numpy.core.multiarray required for multidimensional Array conversions - ", e)
    end
    if pyversion::VersionNumber < v"3.0"
        PyArray_API = @pychecki ccall((@pysym :PyCObject_AsVoidPtr), 
                                      Ptr{Ptr{Void}}, (PyPtr,), 
                                      (npy_multiarray::PyObject)["_ARRAY_API"])
    else
        PyArray_API = @pychecki ccall((@pysym :PyCapsule_GetPointer),
                                      Ptr{Ptr{Void}}, (PyPtr,Ptr{Void}), 
                            (npy_multiarray::PyObject)["_ARRAY_API"], C_NULL)
    end

    # directory for numpy include files to parse
    inc = pycall(pyimport("numpy")["get_include"], String)

    # Parse __multiarray_api.h to obtain length and meaning of PyArray_API
    try
        hdrfile = open(joinpath(inc, "numpy", "__multiarray_api.h"))
        hdr = readall(hdrfile);
        close(hdrfile)
    catch e
        error("could not read __multiarray_api.h to parse PyArray_API ", e)
    end
    hdr = replace(hdr, r"\\\s*\n", " "); # rm backslashed newlines
    r = r"^#define\s+([A-Za-z]\w*)\s+\(.*\bPyArray_API\s*\[\s*([0-9]+)\s*\]\s*\)\s*$"m # regex to match #define PyFoo (... PyArray_API[nnn])
    PyArray_API_length = 0
    for m in eachmatch(r, hdr) # search for max index into PyArray_API 
        PyArray_API_length = max(PyArray_API_length, int(m.captures[2])+1)
    end
    API = pointer_to_array(PyArray_API, (PyArray_API_length,))
    for m in eachmatch(r, hdr) # build npy_api table
        (npy_api::Tnpy_api)[symbol(m.captures[1])] = API[int(m.captures[2])+1]
    end
    if !haskey(npy_api::Tnpy_api, :PyArray_New)
        error("failure parsing NumPy PyArray_API symbol table")
    end

    npy_initialized::Bool = true
    return
end

function npyfinalize()
    global npy_api
    global npy_initialized
    global npy_multiarray

    if npy_initialized::Bool
        empty!(npy_api::Tnpy_api)
        pydecref(npy_multiarray::PyObject)
        npy_initialized::Bool = false
    end
    return
end

#########################################################################
# Hard-coded constant values, copied from numpy/ndarraytypes.h ...
# the values of these seem to have been stable for some time, and
# the NumPy developers seem to have some awareness of binary compatibility

# NPY_TYPES:
const NPY_BOOL = int32(0)
const NPY_BYTE = int32(1)
const NPY_UBYTE = int32(2)
const NPY_SHORT = int32(3)
const NPY_USHORT = int32(4)
const NPY_INT = int32(5)
const NPY_UINT = int32(6)
const NPY_LONG = int32(7)
const NPY_ULONG = int32(8)
const NPY_LONGLONG = int32(9)
const NPY_ULONGLONG = int32(10)
const NPY_FLOAT = int32(11)
const NPY_DOUBLE = int32(12)
const NPY_LONGDOUBLE = int32(13)
const NPY_CFLOAT = int32(14)
const NPY_CDOUBLE = int32(15)
const NPY_CLONGDOUBLE = int32(16)
const NPY_OBJECT = int32(17)
const NPY_STRING = int32(18)
const NPY_UNICODE = int32(19)
const NPY_VOID = int32(20)

# NPY_ORDER:
const NPY_ANYORDER=int32(-1)
const NPY_CORDER=int32(0)
const NPY_FORTRANORDER=int32(1)

# flags:
const NPY_ARRAY_C_CONTIGUOUS = int32(1)
const NPY_ARRAY_F_CONTIGUOUS = int32(2)
const NPY_ARRAY_ALIGNED = int32(0x0100)
const NPY_ARRAY_WRITEABLE = int32(0x0400)
const NPY_ARRAY_OWNDATA = int32(0x0004)
const NPY_ARRAY_ENSURECOPY = int32(0x0020)
const NPY_ARRAY_ENSUREARRAY = int32(0x0040)
const NPY_ARRAY_FORCECAST = int32(0x0010)
const NPY_ARRAY_UPDATEIFCOPY = int32(0x1000)
const NPY_ARRAY_NOTSWAPPED = int32(0x0200)
const NPY_ARRAY_ELEMENTSTRIDES = int32(0x0080)

#########################################################################
# conversion from Julia types to NPY_TYPES constant

npy_type(::Type{Int8}) = NPY_BYTE
npy_type(::Type{Uint8}) = NPY_UBYTE
npy_type(::Type{Int16}) = NPY_SHORT
npy_type(::Type{Uint16}) = NPY_USHORT
npy_type(::Type{Int32}) = NPY_INT
npy_type(::Type{Uint32}) = NPY_UINT
npy_type(::Type{Int64}) = NPY_LONGLONG
npy_type(::Type{Uint64}) = NPY_ULONGLONG
npy_type(::Type{Float32}) = NPY_FLOAT
npy_type(::Type{Float64}) = NPY_DOUBLE
npy_type(::Type{Complex64}) = NPY_CFLOAT
npy_type(::Type{Complex128}) = NPY_CDOUBLE
npy_type(::Type{PyPtr}) = NPY_OBJECT

typealias NPY_TYPES Union(Int8,Uint8,Int16,Uint16,Int32,Uint32,Int64,Uint64,Float32,Float64,Complex64,Complex128,PyPtr)

# conversions from __array_interface__ type strings to supported Julia types
const npy_typestrs = (String=>Type)[ "i1"=>Int8, "u1"=>Uint8,
                                     "i2"=>Int16, "u2"=>Uint16,
                                     "i4"=>Int32, "u4"=>Uint32,
                                     "i8"=>Int64, "u8"=>Uint64,
                                     "f4"=>Float32, "f8"=>Float64,
                                     "c8"=>Complex64, "c16"=>Complex128,
                                     "O$(div(WORD_SIZE,8))"=>PyPtr ]

#########################################################################
# no-copy conversion of Julia arrays to NumPy arrays.

function PyObject{T<:NPY_TYPES}(a::StridedArray{T})
    try
        @npyinitialize
        p = @pychecki ccall(npy_api[:PyArray_New], PyPtr,
              (PyPtr,Cint,Ptr{Int},Cint, Ptr{Int},Ptr{T}, Cint,Cint,PyPtr),
              npy_api[:PyArray_Type], 
              ndims(a), [size(a)...], npy_type(T),
              [strides(a)...] * sizeof(eltype(a)), a, sizeof(eltype(a)),
              NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
              C_NULL)
        return PyObject(p, a)
    catch e
        array2py(a) # fallback to non-NumPy version
    end
end

#########################################################################
# Extract shape and other information about a NumPy array.  We need
# to call the Python interface to do this, since the equivalent information
# in NumPy's C API is only available via macros (or parsing structs).
# [ Hopefully, this will be improved in a future NumPy version. ]

type PyArray_Info
    T::Type
    native::Bool # native byte order?
    sz::Vector{Int}
    st::Vector{Int} # strides, in multiples of bytes!
    data::Ptr{Void}
    readonly::Bool

    function PyArray_Info(a::PyObject)
       # error("could not convert type $T to fmt string")
        ai = PyDict{String,PyObject}(a["__array_interface__"])
        typestr = convert(String, ai["typestr"])
        T = npy_typestrs[typestr[2:end]]
        datatuple = convert((Int,Bool), ai["data"])
        sz = convert(Vector{Int}, ai["shape"])
        local st
        try
            st = isempty(sz) ? Int[] : convert(Vector{Int}, ai["strides"])
        catch
            # default is C-order contiguous 
            st = similar(sz)
            st[end] = sizeof(T)
            for i = length(sz)-1:-1:1
                st[i] = st[i+1]*sz[i+1]
            end
        end
        return new(T,
                   (ENDIAN_BOM == 0x04030201 && typestr[1] == '<')
                   || (ENDIAN_BOM == 0x01020304 && typestr[1] == '>') 
                   || typestr[1] == '|',
                   sz, st,
                   convert(Ptr{Void}, datatuple[1]),
                   datatuple[2])
    end
end

aligned(i::PyArray_Info) = #  FIXME: also check pointer alignment?
  all(m -> m == 0, mod(i.st, sizeof(i.T))) # strides divisible by elsize

# whether a contiguous array in column-major (Fortran, Julia) order
function f_contiguous(T::Type, sz::Vector{Int}, st::Vector{Int})
    if prod(sz) == 1
        return true
    end
    if st[1] != sizeof(T)
        return false
    end
    for j = 2:length(st)
        if st[j] != st[j-1] * sz[j-1]
            return false
        end
    end
    return true
end

f_contiguous(i::PyArray_Info) = f_contiguous(i.T, i.sz, i.st)
c_contiguous(i::PyArray_Info) = f_contiguous(i.T, flipud(i.sz), flipud(i.st))

#########################################################################
# PyArray: no-copy wrapper around NumPy ndarray
#
# Hopefully, in the future this can be a subclass of StridedArray (see
# Julia issue #2345), which will allow it to be used with most Julia
# functions, but that is not possible at the moment.  So, to use this
# with Julia linalg functions etcetera a copy is still required.

type PyArray{T,N} <: AbstractArray{T,N}
    o::PyObject
    info::PyArray_Info
    dims::Dims
    st::Vector{Int}
    f_contig::Bool
    c_contig::Bool
    data::Ptr{T}

    function PyArray(o::PyObject, info::PyArray_Info)
        if !aligned(info)
            throw(ArgumentError("only NPY_ARRAY_ALIGNED arrays are supported"))
        elseif !info.native
	    throw(ArgumentError("only native byte-order arrays are supported"))
        elseif info.T != T
            throw(ArgumentError("inconsistent type in PyArray constructor"))
        elseif length(info.sz) != N || length(info.st) != N
            throw(ArgumentError("inconsistent ndims in PyArray constructor"))
        end
        return new(o, info, tuple(info.sz...), div(info.st, sizeof(T)),
                   f_contiguous(info), c_contiguous(info),
                   convert(Ptr{T}, info.data))
    end
end

function PyArray(o::PyObject)
    info = PyArray_Info(o)
    return PyArray{info.T, length(info.sz)}(o, info)
end

size(a::PyArray) = a.dims
ndims{T,N}(a::PyArray{T,N}) = N

similar(a::PyArray, T, dims::Dims) = Array(T, dims)

function copy{T,N}(a::PyArray{T,N}) 
    if N > 1 && a.c_contig # equivalent to f_contig with reversed dims
        B = pointer_to_array(a.data, ntuple(N, n -> a.dims[N - n + 1]))
        return N == 2 ? transpose(B) : permutedims(B, (N:-1:1))
    end
    A = Array(T, a.dims)
    if a.f_contig
        ccall(:memcpy, Void, (Ptr{T}, Ptr{T}, Int), A, a, sizeof(T)*length(a))
        return A
    else
        return copy!(A, a)
    end
end

# TODO: need to do bounds-checking of these indices!

getindex{T}(a::PyArray{T,0}) = unsafe_load(a.data)
getindex{T}(a::PyArray{T,1}, i::Integer) = unsafe_load(a.data, 1 + (i-1)*a.st[1])

getindex{T}(a::PyArray{T,2}, i::Integer, j::Integer) = 
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

setindex!{T}(a::PyArray{T,0}, v) = writeok_assign(a, v, 1)
setindex!{T}(a::PyArray{T,1}, v, i::Integer) = writeok_assign(a, v, 1 + (i-1)*a.st[1])

setindex!{T}(a::PyArray{T,2}, v, i::Integer, j::Integer) = 
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

convert{T}(::Type{Ptr{T}}, a::PyArray{T}) = a.data

pointer(a::PyArray, i::Int) = pointer(a, ind2sub(a.dims, i))

function pointer{T}(a::PyArray{T}, is::(Int...))
    offset = 0
    for i = 1:length(is)
        offset += (is[i]-1)*a.st[i]
    end
    return a.data + offset*sizeof(T)
end

summary{T}(a::PyArray{T}) = string(Base.dims2string(size(a)), " ",
                                   string(T), " PyArray")

#########################################################################
# PyArray <-> PyObject conversions

PyObject(a::PyArray) = a.o

convert(::Type{PyArray}, o::PyObject) = PyArray(o)

function convert{T<:NPY_TYPES}(::Type{Array{T, 1}}, o::PyObject)
    try
        copy(PyArray{T, 1}(o, PyArray_Info(o))) # will check T and N vs. info
    catch
        len = @pycheckzi ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        A = Array(pyany_toany(T), len)
        py2array(T, A, o, 1, 1)
    end
end

function convert{T<:NPY_TYPES}(::Type{Array{T}}, o::PyObject)
    try
        info = PyArray_Info(o)
        copy(PyArray{T, length(info.sz)}(o, info)) # will check T == info.T
    catch
        py2array(T, o)
    end
end

function convert{T<:NPY_TYPES,N}(::Type{Array{T,N}}, o::PyObject)
    try
        info = PyArray_Info(o)
        copy(PyArray{T,N}(o, info)) # will check T == info.T and N == length(info.sz)
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

function convert{N}(::Type{Array{PyObject,N}}, o::PyObject)
    map(pyincref, convert(Array{PyPtr, N}, o))
end

#########################################################################
