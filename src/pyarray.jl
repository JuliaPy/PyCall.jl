# Support for python buffer protocol backed arrays

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

const pyfmt_jltype = @compat Dict{Char,DataType}(
                                    'x' => Uint8,
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
                                    'P' => Ptr{Void})

#TODO: for now, heterogenous arrays of a single numeric type
const PyBufType = Union(values(pyfmt_jltype)...)

const jltype_pyfmt = [v => k for (k,v) in pyfmt_jltype]

#TODO: this only works for simple struct packing
function parse_pyfmt(buf::PyBuffer)
    # a missing format string is interpreted as a plain byte buffer
    if buf.buf.format == C_NULL
        return (:native, [Uint8,])
    end
    fmt = bytestring(buf.buf.format)
    types = DataType[]
    idx = 1
    c = fmt[idx]
    if c == '@' || c == '='
        byteorder = :native
        idx += 1
    elseif c == '<'
        byteorder = :little
        idx += 1
    elseif c == '>' || c == '!'
        byteorder = :big
        idx += 1
    else
        byteorder = :native
    end
    len = length(fmt)
    while idx <= len
        c = fmt[idx]
        if c == ' ' || c == '\t'
            idx += 1
            continue
        end
        num = 1
        # we punt on overflow checking here for now (num >= C_ssize_t)
        if '0' <= c && c <= '9'
            num = c - '0'
            idx += 1
            if idx > len
                return (byteorder,types)
            end
            c = fmt[idx]
            while '0' <= c && c <= '9'
                num = num * 10 + (c - '0')
                if num < 0
                    throw(OverflowError())
                end
                idx += 1
                if idx > len
                    return (byteorder,types)
                end
                c = fmt[idx]
            end
        end
        ty = get(pyfmt_jltype, c, Void)::DataType
        if ty === Void
            throw(ArgumentError("invalid PyBuffer format string: $fmt"))
        end
        for _ = 1:num
            push!(types,ty)
        end
        idx += 1
    end
    return (byteorder,types)
end

jltype_to_pyfmt{T}(::Type{T}) = jltype_to_pyfmt(IOBuffer(), T)

function jltype_to_pyfmt{T}(io::IO, ::Type{T})
    if nfields(T) == 0
        throw(ArgumentError("no fields for structure type $T"))
    end
    write(io, "T{")
    for n in fieldnames(T)
        ty = fieldtype(T, n)
        if isbits(ty)
            fty = get(jltype_pyfmt, ty, '\0')::Char
            if fty != '\0'
                write(io, "$fty:$n:")
            elseif Base.isstructtype(T)
                jltype_to_pyfmt(io, ty)
            else
                throw(ArgumentError("unknown pyfmt conversion for type $T"))
            end
        else
            throw(ArgumentError("$T is not a bits type"))
        end
    end
    write(io, "}")
    return bytestring(io)
end

pyfmt(b::PyBuffer) = b.buf.format == C_NULL ? "" : bytestring(b.buf.format)

sizeof_pyfmt(fmt::ByteString) = ccall((@pysym :PyBuffer_SizeFromFormat), Cint,
                                      (Ptr{Cchar},), &fmt)

function aligned(b::PyBuffer)
    if b.buf.strides == C_NULL
        # buffer is defined to be C-contiguous
        return true
    end
    for i=1:ndims(b)
        if mod(stride(b,i), b.buf.itemsize) != 0
            return false
        end
    end
    return true
end

f_contiguous(view::PyBuffer) =
    ccall((@pysym :PyBuffer_IsContiguous), Cint,
    	  (Ptr{PyBuffer}, Cchar), &view, 'F') == 1

c_contiguous(view::PyBuffer) =
    ccall((@pysym :PyBuffer_IsContiguous), Cint,
    	  (Ptr{PyBuffer}, Cchar), &view, 'C') == 1

#TODO: PUT READ / WRITE / INDIRECT INTO TYPE PARAMS
type PyArray{T, N} <: DenseArray{T, N}
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
        # TODO
        #elseif eltype(b) != T
	    #    throw(ArgumentError("inconsistent type in PyArray constructor"))
        elseif ndims(b) != N
            throw(ArgumentError("inconsistent ndims in PyArray constructor"))
        end
        return new(o, b, true,
                   Bool(b.buf.readonly),
                   size(b),
                   tuple(Int[div(s,sizeof(T)) for s in strides(b)]...),
                   f_contiguous(b),
                   c_contiguous(b),
                   convert(Ptr{T}, b.buf.buf))
    end
end

function PyArray(o::PyObject)
    view = PyBuffer()
    ret = ccall((@pysym :PyObject_GetBuffer), Cint,
                (PyPtr, Ptr{PyBuffer}, Cint), o, &view, PyBUF_FULL)
    if ret < 0
        # try the readonly buffer interface
        @pycheckzi ccall((@pysym :PyObject_GetBuffer), Cint,
                         (PyPtr, Ptr{PyBuffer}, Cint), o, &view, PyBUF_FULL_RO)
    end
    order, tys = parse_pyfmt(view)
    if order !== :native
        throw(ArgumentError("PyArray cannot yet handle non-native endian buffers"))
    elseif isempty(tys)
        throw(ArgumentError("PyArray cannot yet handle structure types"))
    end
    return PyArray{tys[1], ndims(view)}(o, view)
end

Base.size(a::PyArray) = a.dims
Base.ndims{T,N}(a::PyArray{T,N}) = N
Base.similar(a::PyArray, T, dims::Dims) = Array(T, dims)
Base.stride(a::PyArray, i::Integer) = a.strides[i]
Base.summary{T,N}(a::PyArray{T,N}) =
    string(Base.dims2string(size(a)), " PyArray{$T,$N}")

#TODO: is this correct for all buffer types other than contig/dense?
#TODO: get rid of this, should be copy! but copy! uses similar under the hood
function Base.copy{T,N}(a::PyArray{T,N})
    if N > 1 && a.c_contig i
        # equivalent to f_contig with reversed dims
        B = pointer_to_array(a.data, (Int[a.dims[N - d + 1] for d in 1:N]...))
        if N == 2
            return transpose(B)
        else
            return permutedims(B, N:-1:1)
        end
    end
    A = Array(T, a.dims)
    if a.f_contig
        ccall(:memcpy, Void, (Ptr{T}, Ptr{T}, Int), A, a, sizeof(T) * length(a))
        return A
    end
    return copy!(A, a)
end

Base.getindex{T}(a::PyArray{T,0}) = unsafe_load(a.data)

function Base.getindex{T}(a::PyArray{T,1}, i::Integer)
    1 <= i <= length(a) || throw(BoundsError())
    unsafe_load(a.data, 1 + (i-1) * a.strides[1])
end

function Base.getindex{T}(a::PyArray{T,2}, i::Integer, j::Integer)
    1 <= i <= size(a,1) || throw(BoundsError())
    1 <= j <= size(a,2) || throw(BoundsError())
    unsafe_load(a.data, 1 + (i-1) * a.strides[1] + (j-1) * a.strides[2])
end

function Base.getindex(a::PyArray, i::Integer)
    if a.f_contig
        1 <= i <= length(a) || throw(BoundsError())
        return unsafe_load(a.data, i)
    end
    return getindex(a, ind2sub(a.dims, i)...)
end

function Base.getindex(a::PyArray, is::Integer...)
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
Base.pointer{T}(a::PyArray{T}) = a.data

function Base.pointer{T}(a::PyArray{T}, is::@compat(Tuple{Vararg{Int}}))
    offset = 0
    for i = 1:length(is)
        offset += (is[i] - 1) * a.strides[i]
    end
    return a.data + offset * sizeof(T)
end

function Base.setindex!{T}(a::PyArray{T,0}, v)
    a.readonly && throw(ErrorException("PyArray is read-only"))
    unsafe_store!(pointer(a), v, 1)
    return v
end

function Base.setindex!{T}(a::PyArray{T,1}, v, i::Integer)
    a.readonly && throw(ErrorException("PyArray is read-only"))
    1 <= i <= length(a) || throw(BoundsError())
    unsafe_store!(pointer(a), v, 1 + (i-1) * a.strides[1])
    return v
end

function Base.setindex!{T}(a::PyArray{T,2}, v, i::Integer, j::Integer)
    a.readonly && throw(ErrorException("PyArray is read-only"))
    1 <= i <= size(a,1) || throw(BoundsError())
    1 <= j <= size(a,2) || throw(BoundsError())
    unsafe_store!(pointer(a), v, 1 + (i-1) * a.strides[1] + (j-1) * a.strides[2])
    return v
end

function Base.setindex!(a::PyArray, v, i::Integer)
    a.readonly && throw(ErrorException("PyArray is read-only"))
    if a.f_contig
        1 <= i <= length(a) || throw(BoundsError())
        unsafe_store!(pointer(a), v, i)
        return v
    end
    return setindex!(a, v, ind2sub(a.dims, i)...)
end

function Base.setindex!{T,N}(a::PyArray{T, N}, v, is::Integer...)
    a.readonly && throw(ErrorException("PyArray is read-only"))
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
    unsafe_store!(pointer(a), v, index)
    return v
end

#########################################################################
# PyArray <-> PyObject conversions

PyObject(a::PyArray) = a.o

Base.convert(::Type{PyArray}, o::PyObject) = PyArray(o)

function Base.convert{T<:PyBufType}(::Type{Array{T,1}}, o::PyObject)
    try
        view = PyBuffer(o, PyBUF_RECORDS)
        order, tys = parse_pyfmt(view)
        if length(tys) != 1
            throw(ArgumentError("PyArray cannot yet handle structure types"))
        end
        if tys[1] != T
            throw(ArgumentError("invalid type"))
        end
        ndim = ndims(view)
        if ndim != 1
            throw(ArgumentError("invalid dim"))
        end
        #TODO: return a pyarray and not a copy
        return copy(PyArray{T,1}(o, view))
    catch
        len = @pycheckzi ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        A = Array(pyany_toany(T), len)
        return py2array(T, A, o, 1, 1)
    end
end

Base.convert(::Type{Array{PyObject}}, o::PyObject) =
    map(pyincref, convert(Array{PyPtr}, o))

Base.convert(::Type{Array{PyObject,1}}, o::PyObject) =
    map(pyincref, convert(Array{PyPtr,1}, o))

Base.convert{N}(::Type{Array{PyObject,N}}, o::PyObject) =
    map(pyincref, convert(Array{PyPtr,N}, o))
