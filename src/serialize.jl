using Compat.Serialization
import Compat.Serialization: serialize, deserialize
if VERSION < v"0.7.0-DEV.3476"
    import Base.Serializer: serialize_type
else
    import Compat.Serialization: serialize_type
end

const _pickle = PyNULL()

pickle() = ispynull(_pickle) ? copy!(_pickle, pyimport(PyCall.pyversion.major ≥ 3 ? "pickle" : "cPickle")) : _pickle

function serialize(s::AbstractSerializer, pyo::PyObject)
    serialize_type(s, PyObject)
    if ispynull(pyo)
        serialize(s, pyo.o)
    else
        b = PyBuffer(pycall(pickle()["dumps"], PyObject, pyo))
        serialize(s, unsafe_wrap(Array, Ptr{UInt8}(pointer(b)), sizeof(b)))
    end
end

"""
    pybytes(b::Union{String,DenseVector{UInt8}})

Convert `b` to a Python `bytes` object.   This differs from the default
`PyObject(b)` conversion of `String` to a Python string (which may fail if `b`
does not contain valid Unicode), or from the default conversion of a
`Vector{UInt8}` to a `bytearray` object (which is mutable, unlike `bytes`).
"""
function pybytes(b::Union{String,DenseVector{UInt8}})
    b isa String || stride(b,1) == 1 || throw(ArgumentError("pybytes requires stride-1 byte arrays"))
    PyObject(@pycheckn @pyccall(PyString_FromStringAndSize,
                             PyPtr, (Ptr{UInt8}, Int),
                             b, sizeof(b)))
end

function deserialize(s::AbstractSerializer, t::Type{PyObject})
    b = deserialize(s)
    if isa(b, PyPtr)
        @assert b == C_NULL
        return PyNULL()
    else
        return pycall(pickle()["loads"], PyObject, pybytes(b))
    end
end
