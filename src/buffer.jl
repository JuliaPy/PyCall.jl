# Support for python buffer protocol buffers

# for versions > 3.2 (TODO: obj field not in struct for python versions <= 3.2)

immutable Py_buffer
    buf::Ptr{Void}
    obj::PyPtr
    len::Cssize_t
    itemsize::Cssize_t
    
    readonly::Cint
    ndim::Cint
    format::Ptr{Cchar}
    shape::Ptr{Cssize_t}
    strides::Ptr{Cssize_t}
    suboffsets::Ptr{Cssize_t}
    internal::Ptr{Void}
end
 
type PyBuffer
    buf::Py_buffer

    PyBuffer() = begin
    	b = new(Py_buffer(C_NULL, C_NULL, 0, 0, 
                          0, 0, C_NULL, C_NULL,
                          C_NULL, C_NULL, C_NULL))
        finalizer(b, release!)
        return b 
    end
end

release!(b::PyBuffer) = begin
    if b.buf.obj != C_NULL
        ccall((@pysym :PyBuffer_Release), Void, (Ptr{PyBuffer},), &b)
    end
end

Base.ndims(b::PyBuffer)  = int(b.buf.ndim)
Base.length(b::PyBuffer) = b.buf.ndim >= 1 ? div(b.buf.len, b.buf.itemsize) : 0
Base.sizeof(b::PyBuffer) = b.buf.len

Base.size(b::PyBuffer) = begin
    if b.buf.ndim == 0
        return (0,)
    end
    if b.buf.ndim == 1
        return (div(b.buf.len, b.buf.itemsize),)
    end
    @assert b.buf.shape != C_NULL
    return tuple(Int[unsafe_load(b.buf.shape, i) for i=1:b.buf.ndim]...) 
end 

Base.strides(b::PyBuffer) = begin
    if b.buf.ndim == 0 || b.buf.ndim == 1
        return (1,)
    end
    @assert b.buf.strides != C_NULL
    return tuple(Int[div(unsafe_load(b.buf.strides, i), b.buf.itemsize) for i=1:b.buf.ndim]...)
end

#########################################################################
# hard coded constant values, copied from Cpython's include/object.h

cint(x) = convert(Cint, x)

const PyBUF_MAX_NDIM = cint(64)

const PyBUF_SIMPLE    = cint(0)
const PyBUF_WRITABLE  = cint(0x0001)
const PyBUF_WRITEABLE = PyBUF_WRITABLE
const PyBUF_FORMAT    = cint(0x0004)
const PyBUF_ND        = cint(0x0008)

const PyBUF_STRIDES        = cint(0x0010) | PyBUF_ND
const PyBUF_C_CONTIGUOUS   = cint(0x0020) | PyBUF_STRIDES
const PyBUF_F_CONTIGUOUS   = cint(0x0040) | PyBUF_STRIDES
const PyBUF_ANY_CONTIGUOUS = cint(0x0080) | PyBUF_STRIDES
const PyBUF_INDIRECT       = cint(0x0100) | PyBUF_STRIDES

const PyBUF_CONTIG    = cint(PyBUF_ND | PyBUF_WRITABLE)
const PyBUF_CONTIG_RO = cint(PyBUF_ND)

const PyBUF_STRIDED    = cint(PyBUF_STRIDES | PyBUF_WRITABLE)
const PyBUF_STRIDED_RO = cint(PyBUF_STRIDES)

const PyBUF_RECORDS    = cint(PyBUF_STRIDES | PyBUF_WRITABLE | PyBUF_FORMAT)
const PyBUF_RECORDS_RO = cint(PyBUF_STRIDES | PyBUF_FORMAT)

const PyBUF_FULL    = cint(PyBUF_INDIRECT | PyBUF_WRITABLE | PyBUF_FORMAT)
const PyBUF_FULL_RO = cint(PyBUF_INDIRECT | PyBUF_FORMAT)

const PyBUF_READ  = cint(0x100)
const PyBUF_WRITE = cint(0x200)

########################################################
# parse python's buffer format string
# PEP 3118 format specification
# 't' bit, '4b' specifies # of bits
# '?' platform bool type
# 'g' long double
# 'c' ucs-1 encoding
# 'u' ucs-2 encoding
# 'w' ucs-4 encoding
# 'O' pointer to python object
# 'Z' complex number, Zf (complex float)
# '&' specific pointer, prefix before another character (Ptr{Int8} => '&c')
# 'T{}' structure
# '(k1,k2,...kn) multidim array of whatever follows
# ':name' optional name of preceeding element
# 'X{}' pointer to function (optional function signature placed inside with
#                            return value preceeded by -> @ end)
# X{b:b}->b

const pyfmt_byteorder = (Char => Symbol)['@' => :native,
                                         '=' => :native,
                                         '<' => :little,
                                         '>' => :big,
					                     '!' => :big]

const pyfmt_jltype = (Char => Type)['x' => Uint8,
                                    'c' => Cchar,
                                    'b' => Cuchar,
                                    'B' => Uint8,
                                    '?' => Bool,
                                    'h' => Cshort,
                                    'H' => Cushort,
                                    'i' => Cint,
                                    'I' => Cuint,
                                    'l' => Clong,
                                    'L' => Culong,
                                    'q' => Clonglong,
                                    'Q' => Culonglong,
                                    'n' => Cssize_t,
                                    'N' => Csize_t,
                                    'f' => Float32,
                                    'd' => Float64,
                                    's' => Ptr{Cchar},
                                    'p' => Ptr{Cchar},
                                    'P' => Ptr{Void}]

# for now, heterogenous arrays of a single numeric type
const PyBufType = Union(values(pyfmt_jltype)...)

const jltype_pyfmt = Dict{Type, Char}(collect(values(pyfmt_jltype)),
                                      collect(keys(pyfmt_jltype)))

#TODO: this only works for simple struct packing
function parse_pyfmt(fmt::ByteString)
    types = Type[]
    idx = 1
    byteorder = :native
    if haskey(pyfmt_byteorder, fmt[idx])
        byteorder = pyfmt_byteorder[fmt[idx]]
	    idx = 2
    end
    len = length(fmt)
    while idx <= len
        c = fmt[idx]
        if isblank(c)
            idx += 1
            continue
        end
        num = 1
        # we punt on overflow checking here for now (num >= C_ssize_t)
        if '0' <= c && c <= '9'
            num = c - '0'
            idx += 1
            if idx > len
                return (byteorder, types)
            end
            c = fmt[idx]
            while '0' <= c && c <= '9'
                num = num * 10 + (c - '0')
                idx += 1
                if idx > len
                    return (byteorder, types)
                end
                c = fmt[idx]
            end
        end
        try
            ty = pyfmt_jltype[c]
            for _ = 1:num
                push!(types, ty)
            end
        catch
            throw(ArgumentError("invalid PyBuffer format string $fmt"))
        end
        idx += 1
    end
    return (byteorder, types)
end

jltype_to_pyfmt{T}(::Type{T}) = jltype_to_pyfmt(IOBuffer(), T)

function jltype_to_pyfmt{T}(io::IO, ::Type{T})
    length(T.names) == 0 && error("no fields for structure type $T") 
    write(io, "T{")
    idx = 1
    for n in T.names
        ty = T.types[idx]
        if isbits(ty)
            if haskey(jltype_pyfmt, ty)
                fty = jltype_pyfmt[ty]
                write(io, "$fty:$n:")
            elseif Base.isstructtype(T)
                jltype_to_pyfmt(io, ty)
            else
                error("pyfmt unknown conversion for type $T")
            end
        else
            error("pyfmt can only encode bits types")
        end
        idx += 1
    end
    write(io, "}")
    return bytestring(io)
end

pyfmt(b::PyBuffer) = b.buf.format == C_NULL ? bytestring("") : bytestring(b.buf.format)

sizeof_pyfmt(fmt::ByteString) = ccall((@pysym :PyBuffer_SizeFromFormat), Cint,
                                      (Ptr{Cchar},), &fmt)

#########################################################################

pygetbuffer(o::PyObject, flags::Cint) = begin
    view = PyBuffer()  
    @pycheckzi ccall((@pysym :PyObject_GetBuffer), Cint, 
    	             (PyPtr, Ptr{PyBuffer}, Cint),
		             o.o, &view, flags)
    return view
end

aligned(b::PyBuffer) = begin
    if b.buf.strides == C_NULL
        throw(ArgumentError("PyBuffer strides field is NULL"))
    end
    for i=1:b.buf.ndim
        if mod(unsafe_load(b.buf.strides, i), b.buf.itemsize) != 0
	        return false
        end
    end
    return true
end

f_contiguous(view::PyBuffer) = 
    ccall((@pysym :PyBuffer_IsContiguous), Cint, 
    	  (Ptr{PyBuffer}, Cchar), &view, 'F') == cint(1)

c_contiguous(view::PyBuffer) =
    ccall((@pysym :PyBuffer_IsContiguous), Cint,
    	  (Ptr{PyBuffer}, Cchar), &view, 'C') == cint(1)

iscontiguous(view::PyBuffer) = 
    ccall((@pysym :PyBuffer_IsContiguous), Cint,
    	  (Ptr{PyBuffer}, Cchar), &view, 'A') == cint(1)

type PyArray{T, N} <: AbstractArray{T, N}
    o::PyObject
    buf::PyBuffer
    native::Bool
    readonly::Bool
    dims::Dims
    strides::Dims
    f_contig::Bool
    c_contig::Bool
    data::Ptr{T}

    function PyArray(o::PyObject, b::PyBuffer)
        if !aligned(b)
	    throw(ArgumentError("only aligned buffer objects are supported"))
	    throw(ArgumentError("inconsistent type in PyArray constructor"))
        elseif ndims(b) != N
            throw(ArgumentError("inconsistent ndims in PyArray constructor"))
        end
        return new(o, b, true, bool(b.buf.readonly),
                   size(b), strides(b), 
                   f_contiguous(b), 
                   c_contiguous(b),
                   convert(Ptr{T}, b.buf.buf))
    end
end

function PyArray(o::PyObject)
    view = pygetbuffer(o, PyBUF_RECORDS)
    view.buf.format == C_NULL && error("buffer has no format string")
    order, tys = parse_pyfmt(bytestring(view.buf.format))
    length(tys) != 1 && error("PyArray cannot yet handle structure types")
    ty   = tys[1]
    ndim = ndims(view)
    return PyArray{ty, ndim}(o, view)
end

Base.size(a::PyArray) = a.dims
Base.ndims{T,N}(a::PyArray{T,N}) = N
Base.similar(a::PyArray, T, dims::Dims) = Array(T, dims)
Base.stride(a::PyArray, i::Integer) = a.strides[i]
Base.convert{T}(::Type{Ptr{T}}, a::PyArray{T}) = a.data

Base.summary{T}(a::PyArray{T}) = string(Base.dims2string(size(a)), " ",
                                        string(T), " PyArray")

#TODO: is this correct for all buffer types other than contig/dense?
Base.copy{T,N}(a::PyArray{T,N}) = begin
    if N > 1 && a.c_contig # equivalent to f_contig with reversed dims
        B = pointer_to_array(a.data, ntuple(N, n -> a.dims[N - n + 1]))
        return N == 2 ? transpose(B) : permutedims(B, (N:-1:1))
    end
    A = Array(T, a.dims)
    if a.f_contig
        ccall(:memcpy, Void, (Ptr{T}, Ptr{T}, Int), A, a, sizeof(T) * length(a))
        return A
    else
        return copy!(A, a)
    end
end

#TODO: Bounds checking is needed
Base.getindex{T}(a::PyArray{T,0}) = unsafe_load(a.data)

Base.getindex{T}(a::PyArray{T,1}, i::Integer) = 
    unsafe_load(a.data, 1 + (i-1) * a.strides[1])

Base.getindex{T}(a::PyArray{T,2}, i::Integer, j::Integer) = 
    unsafe_load(a.data, 1 + (i-1) * a.strides[1] + (j-1) * a.strides[2])

Base.getindex(a::PyArray, i::Integer) = begin
    if a.f_contig
        return unsafe_load(a.data, i)
    else
        return a[ind2sub(a.dims, i)...]
    end
end

Base.getindex(a::PyArray, is::Integer...) = begin
    index = 1
    n = min(length(is), length(a.strides))
    for i = 1:n
        index += (is[i] - 1) * a.strides[i]
    end
    for i = n+1:length(is)
        if is[i] != 1
            throw(BoundsError())
        end
    end
    return unsafe_load(a.data, index)
end

#TODO: This is only correct for dense, contiguous buffers
Base.pointer{T}(a::PyArray{T}, is::(Int...)) = begin
    offset = 0
    for i = 1:length(is)
        offset += (is[i] - 1) * a.strides[i]
    end
    return a.data + offset * sizeof(T)
end

function writeok_assign(a::PyArray, v, i::Integer)
    a.readonly && throw(ArgumentError("read-only PyArray"))
    unsafe_store!(a.data, v, i)
    return a
end

Base.setindex!{T}(a::PyArray{T,0}, v) = writeok_assign(a, v, 1)

Base.setindex!{T}(a::PyArray{T,1}, v, i::Integer) = 
    writeok_assign(a, v, 1 + (i-1) * a.strides[1])

Base.setindex!{T}(a::PyArray{T,2}, v, i::Integer, j::Integer) = 
    writeok_assign(a, v, 1 + (i-1) * a.strides[1] + (j-1) * a.strides[2])

Base.setindex!(a::PyArray, v, i::Integer) = begin
    if a.f_contig
        return writeok_assign(a, v, i)
    else
        return setindex!(a, v, ind2sub(a.dims, i)...)
    end
end

Base.setindex!{T,N}(a::PyArray{T, N}, v, is::Integer...) = begin
    index = 1
    n = min(length(is), N)
    for i = 1:n
        index += (is[i] - 1) * a.strides[i]
    end
    for i = n+1:length(is)
        if is[i] != 1
            throw(BoundsError())
        end
    end
    return writeok_assign(a, v, index)
end

#########################################################################
# PyArray <-> PyObject conversions

PyObject(a::PyArray) = a.o

Base.convert(::Type{PyArray}, o::PyObject) = PyArray(o)

Base.convert{T<:PyBufType}(::Type{Array{T, 1}}, o::PyObject) = begin
    try
        view = pygetbuffer(o, PyBUF_RECORDS)
        view.format == C_NULL && error("buffer has no format string")
        order, tys = parse_pyfmt(bytestring(view.format))
        length(tys) != 1 && error("PyArray cannot yet handle structure types")
        tys[1] != T && error("invalid type")
        ndim = ndims(view)
        ndim != 1 && error("invalid dim")	
        return copy(PyArray{T, 1}(o, view))
    catch
        len = @pycheckzi ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        A = Array(pyany_toany(T), len)
        return py2array(T, A, o, 1, 1)
    end
end

Base.convert(::Type{Array{PyObject}}, o::PyObject) =
    map(pyincref, convert(Array{PyPtr}, o))

Base.convert(::Type{Array{PyObject,1}}, o::PyObject) =
    map(pyincref, convert(Array{PyPtr, 1}, o))

Base.convert{N}(::Type{Array{PyObject,N}}, o::PyObject) =
    map(pyincref, convert(Array{PyPtr, N}, o))
