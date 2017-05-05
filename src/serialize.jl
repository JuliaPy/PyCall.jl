const _pickle = PyNULL()

function pickle()
    if _pickle.o == C_NULL
        if pyexists(:cPickle)
            copy!(_pickle, pyimport(:cPickle))
        else
            copy!(_pickle, pyimport(:pickle))
        end
    end

    return _pickle
end

function Base.serialize(s::AbstractSerializer, pyo::PyObject)
    Base.serialize_type(s, PyObject)
    b = PyBuffer(pycall(pickle()["dumps"], PyObject, pyo))
    serialize(s, unsafe_wrap(Array, Ptr{UInt8}(pointer(b)), sizeof(b)))
end

function Base.deserialize(s::AbstractSerializer, t::Type{PyObject})
    b = deserialize(s)

    pycall(pickle()["loads"], PyObject,
        PyObject(@pycheckn ccall(@pysym(PyString_FromStringAndSize),
                PyPtr, (Ptr{UInt8}, Int),
                b, sizeof(b))))
end
