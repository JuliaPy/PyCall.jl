# Support Python iterators in Julia and vice versa

#########################################################################
# Iterating over Python objects in Julia

function start(po::PyObject)
    sigatomic_begin()
    try
        o = PyObject(@pycheckn ccall((@pysym :PyObject_GetIter), PyPtr, (PyPtr,), po))
        nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), o))

        return (nxt,o)
    finally
        sigatomic_end()
    end
end

function next(po::PyObject, s)
    sigatomic_begin()
    try
        nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), s[2]))
        return (convert(PyAny, s[1]), (nxt, s[2]))
    finally
        sigatomic_end()
    end
end

done(po::PyObject, s) = ispynull(s[1])

# issue #216
function Base.collect(::Type{T}, o::PyObject) where T
    a = T[]
    for x in o
        push!(a, x)
    end
    return a
end
Base.collect(o::PyObject) = collect(Any, o)

#########################################################################
# Iterating over Julia objects in Python

const jlWrapIteratorType = PyTypeObject()

# Given a Julia object o, return a jlwrap_iterator Python iterator object
# that wraps the Julia iteration protocol with the Python iteration protocol.
# Internally, the jlwrap_iterator object stores the tuple (o, Ref(start(o))),
# where the Ref is used to store the iterator state (which updates during iteration).
function jlwrap_iterator(o::Any)
    if jlWrapIteratorType.tp_name == C_NULL # lazily initialize
        pyjlwrap_type!(jlWrapIteratorType, "PyCall.jlwrap_iterator") do t
            t.tp_iter = cfunction(pyincref_, PyPtr, Tuple{PyPtr}) # new reference to same object
            t.tp_iternext = cfunction(pyjlwrap_iternext, PyPtr, Tuple{PyPtr})
        end
    end
    return pyjlwrap_new(jlWrapIteratorType, (o, Ref(start(o))))
end

# tp_iternext object of a jlwrap_iterator object, similar to PyIter_Next
function pyjlwrap_iternext(self_::PyPtr)
    try
        iter, stateref = unsafe_pyjlwrap_to_objref(self_)
        state = stateref[]
        if !done(iter, state)
            item, state′ = next(iter, state)
            stateref[] = state′ # stores new state in the iterator object
            return pystealref!(PyObject(item))
        end
    catch e
        pyraise(e)
    end
    return PyPtr_NULL
end

# the tp_iter slot of jlwrap object: like PyObject_GetIter, it
# returns a reference to a new jlwrap_iterator object
function pyjlwrap_getiter(self_::PyPtr)
    try
        self = unsafe_pyjlwrap_to_objref(self_)
        return pystealref!(jlwrap_iterator(self))
    catch e
        pyraise(e)
    end
    return PyPtr_NULL
end
