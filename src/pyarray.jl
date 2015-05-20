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

const pyfmt_byteorder = @compat Dict{Char,Symbol}('@' => :native,
                                         '=' => :native,
                                         '<' => :little,
                                         '>' => :big,
					                     '!' => :big)

const pyfmt_jltype = @compat Dict{Char,Type}('x' => Uint8,
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

# for now, heterogenous arrays of a single numeric type
const PyBufType = Union(values(pyfmt_jltype)...)

const jltype_pyfmt = [v => k for (k,v) in pyfmt_jltype]

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
    nfields(T) == 0 && error("no fields for structure type $T")
    write(io, "T{")
    for n in fieldnames(T)
        ty = fieldtype(T, n)
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
    end
    write(io, "}")
    return bytestring(io)
end

pyfmt(b::PyBuffer) = b.buf.format == C_NULL ? bytestring("") : bytestring(b.buf.format)

sizeof_pyfmt(fmt::ByteString) = ccall((@pysym :PyBuffer_SizeFromFormat), Cint,
                                      (Ptr{Cchar},), &fmt)

function aligned(b::PyBuffer)
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
    #TODO: PyBUF_INDIRECT, READONLY
    view = PyBuffer(o, PyBUF_RECORDS)
    if view.buf.format == C_NULL
        throw(ArgumentError("Python buffer has no format string"))
    end
    order, tys = parse_pyfmt(bytestring(view.buf.format))
    if isempty(tys)
        throw(ArgumentError("PyArray cannot yet handle structure types"))
    end
    ty   = first(tys)
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
#TODO: get rid of this, should be copy! but copy! uses similar under the hood
function Base.copy{T,N}(a::PyArray{T,N})
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

function Base.getindex{T}(a::PyArray{T,1}, i::Integer)
    unsafe_load(a.data, 1 + (i-1) * a.strides[1])
end

function Base.getindex{T}(a::PyArray{T,2}, i::Integer, j::Integer)
    unsafe_load(a.data, 1 + (i-1) * a.strides[1] + (j-1) * a.strides[2])
end

function Base.getindex(a::PyArray, i::Integer)
    if a.f_contig
        return unsafe_load(a.data, i)
    else
        return a[ind2sub(a.dims, i)...]
    end
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
function Base.pointer{T}(a::PyArray{T}, is::@compat(Tuple{Vararg{Int}}))
    offset = 0
    for i = 1:length(is)
        offset += (is[i] - 1) * a.strides[i]
    end
    return a.data + offset * sizeof(T)
end

#TODO: This would not be a defined method when PyArray has Read / Write in the type parameter
function writeok_assign(a::PyArray, v, i::Integer)
    unsafe_store!(a.data, v, i)
    v
end

Base.setindex!{T}(a::PyArray{T,0}, v) = writeok_assign(a, v, 1)

Base.setindex!{T}(a::PyArray{T,1}, v, i::Integer) =
    writeok_assign(a, v, 1 + (i-1) * a.strides[1])

Base.setindex!{T}(a::PyArray{T,2}, v, i::Integer, j::Integer) =
    writeok_assign(a, v, 1 + (i-1) * a.strides[1] + (j-1) * a.strides[2])

function Base.setindex!(a::PyArray, v, i::Integer)
    if a.f_contig
        return writeok_assign(a, v, i)
    else
        return setindex!(a, v, ind2sub(a.dims, i)...)
    end
end

function Base.setindex!{T,N}(a::PyArray{T, N}, v, is::Integer...)
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

function Base.convert{T<:PyBufType}(::Type{Array{T,1}}, o::PyObject)
    try
        view = PyBuffer(o, PyBUF_RECORDS)
        order, tys = parse_pyfmt(bytestring(view.format))
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
    map(pyincref, convert(Array{PyPtr, 1}, o))

Base.convert{N}(::Type{Array{PyObject,N}}, o::PyObject) =
    map(pyincref, convert(Array{PyPtr, N}, o))
