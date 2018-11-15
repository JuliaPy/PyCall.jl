# Python buffer protocol: this is a NumPy-array-like facility to get raw
# pointers to contiguous buffers of data underlying other objects,
# with support for describing multiple dimensions, strides, etc.
#     (thanks to @jakebolewski for his work on this)

#############################################################################
# mirror of Py_buffer struct in Python Include/object.h

struct Py_buffer
    buf::Ptr{Cvoid}
    obj::PyPtr
    len::Cssize_t
    itemsize::Cssize_t

    readonly::Cint
    ndim::Cint
    format::Ptr{Cchar}
    shape::Ptr{Cssize_t}
    strides::Ptr{Cssize_t}
    suboffsets::Ptr{Cssize_t}

    # some opaque padding fields to account for differences between
    # Python versions (the structure changed in Python 2.7 and 3.3)
    internal0::Ptr{Cvoid}
    internal1::Ptr{Cvoid}
    internal2::Ptr{Cvoid}
end

mutable struct PyBuffer
    buf::Py_buffer
    PyBuffer() = begin
        b = new(Py_buffer(C_NULL, PyPtr_NULL, 0, 0,
                          0, 0, C_NULL, C_NULL, C_NULL, C_NULL,
                          C_NULL, C_NULL, C_NULL))
        @compat finalizer(pydecref, b)
        return b
    end
end

"""
`pydecref(o::PyBuffer)`
Release the reference to buffer `o`
N.b. As per https://docs.python.org/3/c-api/buffer.html#c.PyBuffer_Release,
It is an error to call this function on a PyBuffer that was not obtained via
the python c-api function `PyObject_GetBuffer()`, unless o.obj is a PyPtr(C_NULL)
"""
function pydecref(o::PyBuffer)
    # note that PyBuffer_Release sets o.obj to NULL, and
    # is a no-op if o.obj is already NULL
    # TODO change to `Ref{PyBuffer}` when 0.6 is dropped.
    _finalized[] || ccall(@pysym(:PyBuffer_Release), Cvoid, (Any,), o)
    o
end

#############################################################################
# Array-like accessors for PyBuffer.

Base.ndims(b::PyBuffer) = UInt(b.buf.ndim)

# from the Python docs: If shape is NULL as a result of a PyBUF_SIMPLE
# or a PyBUF_WRITABLE request, the consumer must disregard itemsize
# and assume itemsize == 1.
Base.length(b::PyBuffer) = b.buf.shape == C_NULL ? Int(b.buf.len) : Int(div(b.buf.len, b.buf.itemsize))

Base.sizeof(b::PyBuffer) = Int(b.buf.len)
Base.pointer(b::PyBuffer) = b.buf.buf

function Base.size(b::PyBuffer)
    b.buf.ndim <= 1 && return (length(b),)
    @assert b.buf.shape != C_NULL
    return tuple(Int[unsafe_load(b.buf.shape, i) for i=1:b.buf.ndim]...)
end
# specialize size(b, d) for efficiency (avoid tuple construction)
function Base.size(b::PyBuffer, d::Integer)
    d > b.buf.ndim && return 1
    d < 0 && throw(BoundsError())
    b.buf.ndim <= 1 && return length(b)
    @assert b.buf.shape != C_NULL
    return Int(unsafe_load(b.buf.shape, d))
end

# stride in bytes for i-th dimension
function Base.stride(b::PyBuffer, d::Integer)
    d > b.buf.ndim && return length(b) # as in base
    d < 0 && throw(BoundsError())
    if b.buf.strides == C_NULL
        if b.buf.ndim == 1
            return b.buf.shape == C_NULL ? 1 : Int(b.buf.itemsize)
        else
            error("unknown buffer strides")
        end
    end
    return Int(unsafe_load(b.buf.strides, d))
end

# Strides in bytes
Base.strides(b::PyBuffer) = ((stride(b,i) for i in 1:b.buf.ndim)...,)

# TODO change to `Ref{PyBuffer}` when 0.6 is dropped.
iscontiguous(b::PyBuffer) =
    1 == ccall((@pysym :PyBuffer_IsContiguous), Cint,
               (Any, Cchar), b, 'A')

#############################################################################
# pybuffer constant values from Include/object.h
const PyBUF_SIMPLE    = convert(Cint, 0)
const PyBUF_WRITABLE  = convert(Cint, 0x0001)
const PyBUF_FORMAT    = convert(Cint, 0x0004)
const PyBUF_ND        = convert(Cint, 0x0008)
const PyBUF_STRIDES        = convert(Cint, 0x0010) | PyBUF_ND
const PyBUF_C_CONTIGUOUS   = convert(Cint, 0x0020) | PyBUF_STRIDES
const PyBUF_F_CONTIGUOUS   = convert(Cint, 0x0040) | PyBUF_STRIDES
const PyBUF_ANY_CONTIGUOUS = convert(Cint, 0x0080) | PyBUF_STRIDES
const PyBUF_INDIRECT       = convert(Cint, 0x0100) | PyBUF_STRIDES
const PyBUF_ND_CONTIGUOUS = Cint(PyBUF_WRITABLE | PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES | PyBUF_ANY_CONTIGUOUS)

# construct a PyBuffer from a PyObject, if possible
function PyBuffer(o::Union{PyObject,PyPtr}, flags=PyBUF_SIMPLE)
    return PyBuffer!(PyBuffer(), o, flags)
end

function PyBuffer!(b::PyBuffer, o::Union{PyObject,PyPtr}, flags=PyBUF_SIMPLE)
    # TODO change to `Ref{PyBuffer}` when 0.6 is dropped.
    pydecref(b) # ensure b is properly released
    @pycheckz ccall((@pysym :PyObject_GetBuffer), Cint,
                     (PyPtr, Any, Cint), o, b, flags)
    return b
end

"""
`isbuftype(b::PyBuffer, o::Union{PyObject,PyPtr}, flags=PyBUF_ND_CONTIGUOUS)`
Returns true if the python object `o` supports the buffer protocol. False if not.
"""
function isbuftype(o::Union{PyObject,PyPtr})
    # PyObject_CheckBuffer is defined in a header file here: https://github.com/python/cpython/blob/ef5ce884a41c8553a7eff66ebace908c1dcc1f89/Include/abstract.h#L510
    # so we can't access it easily. It basically just checks if PyObject_GetBuffer exists
    # So we'll just try call PyObject_GetBuffer and check for success/failure
    b = PyBuffer()
    ret = ccall((@pysym :PyObject_GetBuffer), Cint,
                     (PyPtr, Any, Cint), o, b, PyBUF_ND_CONTIGUOUS)
    if ret != 0
        pyerr_clear()
    else
        # handle pointer types
        T, native_byteorder = array_format(b)
        T <: Ptr && (ret = 1)
    end
    return ret == 0
end

#############################################################################

# recursive function to write buffer dimension by dimension, starting at
# dimension d with the given pointer offset (in bytes).
function writedims(io::IO, b::PyBuffer, offset, d)
    n = 0
    s = stride(b, d)
    if d < b.buf.ndim
        for i = 1:size(b,d)
            n += writedims(io, b, offset, d+1)
            offset += s
        end
    else
        @assert d == b.buf.ndim
        p = convert(Ptr{UInt8}, pointer(b)) + offset
        for i = 1:size(b,d)
            # would be nicer not to write this one byte at a time,
            # but the alternative seems to be to create an Array
            # object on each loop iteration.
            for j = 1:b.buf.itemsize
                n += write(io, unsafe_load(p))
                p += 1
            end
            p += s
        end
    end
    return n
end

function Base.write(io::IO, b::PyBuffer)
    b.buf.obj != C_NULL || error("attempted to write NULL buffer")

    if iscontiguous(b)
        # (note that 0-dimensional buffers are always contiguous)
        return write(io, pointer_to_array(convert(Ptr{UInt8}, pointer(b)),
                                          sizeof(b)))
    else
        return writedims(io, b, 0, 1)
    end
end

# ref: https://github.com/numpy/numpy/blob/v1.14.2/numpy/core/src/multiarray/buffer.c#L966

const standard_typestrs = Dict{String,DataType}(
                           "?"=>Bool,        "P"=>Ptr{Cvoid},
                           "b"=>Int8,        "B"=>UInt8,
                           "h"=>Int16,       "H"=>UInt16,
                           "i"=>Int32,       "I"=>UInt32,
                           "l"=>Int32,       "L"=>UInt32,
                           "q"=>Int64,       "Q"=>UInt64,
                           "e"=>Float16,     "f"=>Float32,
                           "d"=>Float64,     "g"=>Nothing, # Float128?
                           # `Nothing` indicates no equiv Julia type
                           "Z8"=>ComplexF32, "Z16"=>ComplexF64,
                           "Zf"=>ComplexF32, "Zd"=>ComplexF64)

const native_typestrs = Dict{String,DataType}(
                           "?"=>Bool,        "P"=>Ptr{Cvoid},
                           "b"=>Int8,        "B"=>UInt8,
                           "h"=>Cshort,      "H"=>Cushort,
                           "i"=>Cint,        "I"=>Cuint,
                           "l"=>Clong,       "L"=>Culong,
                           "q"=>Clonglong,   "Q"=>Culonglong,
                           "e"=>Float16,     "f"=>Cfloat,
                           "d"=>Cdouble,     "g"=>Nothing, # Float128?
                           # `Nothing` indicates no equiv Julia type
                           "Z8"=>ComplexF32, "Z16"=>ComplexF64,
                           "Zf"=>ComplexF32, "Zd"=>ComplexF64)

const typestrs_native =
    Dict{DataType, String}(zip(values(native_typestrs), keys(native_typestrs)))

get_format_str(pybuf::PyBuffer) = unsafe_string(convert(Ptr{UInt8}, pybuf.buf.format))

function array_format(pybuf::PyBuffer)
    # a NULL-terminated format-string ... indicating what is in each element of memory.
    # TODO: handle more cases: https://www.python.org/dev/peps/pep-3118/#additions-to-the-struct-string-syntax
    # refs: https://github.com/numpy/numpy/blob/v1.14.2/numpy/core/src/multiarray/buffer.c#L966
    #       https://github.com/numpy/numpy/blob/v1.14.2/numpy/core/_internal.py#L490
    #       https://docs.python.org/2/library/struct.html#byte-order-size-and-alignment

    # "NULL implies standard unsigned bytes ("B")" --pep 3118
    pybuf.buf.format == C_NULL && return UInt8, true

    fmt_str = get_format_str(pybuf)
    native_byteorder = true
    type_start_idx = 1
    typestrs = standard_typestrs
    if length(fmt_str) > 1
        type_start_idx = 2
        if fmt_str[1] == '='
        elseif fmt_str[1] == '<'
            native_byteorder = ENDIAN_BOM == 0x04030201
        elseif fmt_str[1] == '>' || fmt_str =='!'
            native_byteorder = ENDIAN_BOM == 0x01020304
        elseif fmt_str[1] == '@' || fmt_str[1] == '^'
            typestrs = native_typestrs
        elseif fmt_str[1] == "Z"
            type_start_idx = 1
        else
            error("Unsupported format string: \"$fmt_str\"")
        end
    end
    typestrs[fmt_str[type_start_idx:end]], native_byteorder
end

#############################################################################
