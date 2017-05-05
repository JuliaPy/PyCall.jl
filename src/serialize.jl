const _pickle = PyNULL()

pickle() = _pickle.o == C_NULL ? copy!(_pickle, pyimport(PyCall.pyversion.major ≥ 3 ? "pickle" : "cPickle")) : _pickle

function Base.serialize(s::AbstractSerializer, pyo::PyObject)
    pyo.o == C_NULL && return invoke(serialize, (AbstractSerializer, Any), s, pyo)
    Base.serialize_type(s, PyObject)
    b = PyBuffer(pycall(pickle()["dumps"], PyObject, pyo))
    serialize(s, unsafe_wrap(Array, Ptr{UInt8}(pointer(b)), sizeof(b)))
end

function Base.deserialize(s::AbstractSerializer, t::Type{PyObject})
    b = deserialize(s)
    return isa(b, PyPtr) ? PyObject(b) : pycall(pickle()["loads"], PyObject,
        PyObject(@pycheckn ccall(@pysym(PyString_FromStringAndSize),
                PyPtr, (Ptr{UInt8}, Int),
                b, sizeof(b))))
end
