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
        b = new(Py_buffer(C_NULL, C_NULL, 0, 0,
                          0, 0, C_NULL, C_NULL, C_NULL, C_NULL,
                          C_NULL, C_NULL, C_NULL))
        @compat finalizer(pydecref, b)
        return b
    end
end

function pydecref(o::PyBuffer)
    # note that PyBuffer_Release sets o.obj to NULL, and
    # is a no-op if o.obj is already NULL
    # TODO change to `Ref{PyBuffer}` when 0.6 is dropped.
    ccall(@pysym(:PyBuffer_Release), Cvoid, (Any,), o)
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

# TODO change to `Ref{PyBuffer}` when 0.6 is dropped.
iscontiguous(b::PyBuffer) =
    1 == ccall((@pysym :PyBuffer_IsContiguous), Cint,
               (Any, Cchar), b, 'A')

#############################################################################
# pybuffer constant values from Include/object.h
const PyBUF_MAX_NDIM = convert(Cint, 64)
const PyBUF_SIMPLE    = convert(Cint, 0)
const PyBUF_WRITABLE  = convert(Cint, 0x0001)
const PyBUF_FORMAT    = convert(Cint, 0x0004)
const PyBUF_ND        = convert(Cint, 0x0008)
const PyBUF_STRIDES        = convert(Cint, 0x0010) | PyBUF_ND
const PyBUF_C_CONTIGUOUS   = convert(Cint, 0x0020) | PyBUF_STRIDES
const PyBUF_F_CONTIGUOUS   = convert(Cint, 0x0040) | PyBUF_STRIDES
const PyBUF_ANY_CONTIGUOUS = convert(Cint, 0x0080) | PyBUF_STRIDES
const PyBUF_INDIRECT       = convert(Cint, 0x0100) | PyBUF_STRIDES

# construct a PyBuffer from a PyObject, if possible
function PyBuffer(o::Union{PyObject,PyPtr}, flags=PyBUF_SIMPLE)
    b = PyBuffer()
    # TODO change to `Ref{PyBuffer}` when 0.6 is dropped.
    @pycheckz ccall((@pysym :PyObject_GetBuffer), Cint,
                     (PyPtr, Any, Cint), o, b, flags)
    return b
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

#############################################################################
