const pickle = PyNULL()

function get_pickle()
    if pickle.o == C_NULL
        copy!(pickle, pyimport("pickle"))
    end

    return pickle
end

function Base.serialize(s::AbstractSerializer, pyo::PyObject)
    Base.serialize_type(s, PyObject)
    serialize(s, get_pickle()[:dumps](pyo))
end

function Base.deserialize(s::AbstractSerializer, t::Type{PyObject})
    get_pickle()[:loads](deserialize(s))
end
