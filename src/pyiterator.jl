# Support Python iterators in Julia and vice versa

#########################################################################
# Iterating over Python objects in Julia

function _start(po::PyObject)
    sigatomic_begin()
    try
        o = PyObject(@pycheckn ccall((@pysym :PyObject_GetIter), PyPtr, (PyPtr,), po))
        nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), o))

        return (nxt,o)
    finally
        sigatomic_end()
    end
end
@static if VERSION < v"0.7.0-DEV.5126" # julia#25261
    Base.start(po::PyObject) = _start(po)

    function Base.next(po::PyObject, s)
        sigatomic_begin()
        try
            nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), s[2]))
            return (convert(PyAny, s[1]), (nxt, s[2]))
        finally
            sigatomic_end()
        end
    end

    Base.done(po::PyObject, s) = ispynull(s[1])
else
    function Base.iterate(po::PyObject, s=_start(po))
        ispynull(s[1]) && return nothing
        sigatomic_begin()
        try
            nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), s[2]))
            return (convert(PyAny, s[1]), (nxt, s[2]))
        finally
            sigatomic_end()
        end
    end
end

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

# tp_iternext object of a jlwrap_iterator object, similar to PyIter_Next
@static if VERSION < v"0.7.0-DEV.5126" # julia#25261
    function pyjlwrap_iternext(self_::PyPtr)
        try
            iter, stateref = unsafe_pyjlwrap_to_objref(self_)
            state = stateref[]
            if !done(iter, state)
                item, state′ = next(iter, state)
                stateref[] = state′ # stores new state in the iterator object
                return pyreturn(item)
            end
        catch e
            pyraise(e)
        end
        return PyPtr_NULL
    end
else
    function pyjlwrap_iternext(self_::PyPtr)
        try
            iter, iter_result_ref = unsafe_pyjlwrap_to_objref(self_)
            iter_result = iter_result_ref[]
            if iter_result !== nothing
                item, state = iter_result
                iter_result_ref[] = iterate(iter, state)
                return pyreturn(item)
            end
        catch e
            pyraise(e)
        end
        return PyPtr_NULL
    end
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

@static if VERSION < v"0.7.0-DEV.5126" # julia#25261
    # Given a Julia object o, return a jlwrap_iterator Python iterator object
    # that wraps the Julia iteration protocol with the Python iteration protocol.
    # Internally, the jlwrap_iterator object stores the tuple (o, Ref(start(o))),
    # where the Ref is used to store the iterator state (which updates during iteration).
    function jlwrap_iterator(o::Any)
        if jlWrapIteratorType.tp_name == C_NULL # lazily initialize
            pyjlwrap_type!(jlWrapIteratorType, "PyCall.jlwrap_iterator") do t
                t.tp_iter = @cfunction(pyincref_, PyPtr, (PyPtr,)) # new reference to same object
                t.tp_iternext = @cfunction(pyjlwrap_iternext, PyPtr, (PyPtr,))
            end
        end
        return pyjlwrap_new(jlWrapIteratorType, (o, Ref(start(o))))
    end
else
    # Given a Julia object o, return a jlwrap_iterator Python iterator object
    # that wraps the Julia iteration protocol with the Python iteration protocol.
    # Internally, the jlwrap_iterator object stores the tuple (o, Ref(iterate(o))),
    # where the Ref is used to store the (element, state)-iterator result tuple
    # (which updates during iteration) and also the nothing termination indicator.
    function jlwrap_iterator(o::Any)
        if jlWrapIteratorType.tp_name == C_NULL # lazily initialize
            pyjlwrap_type!(jlWrapIteratorType, "PyCall.jlwrap_iterator") do t
                t.tp_iter = @cfunction(pyincref_, PyPtr, (PyPtr,)) # new reference to same object
                t.tp_iternext = @cfunction(pyjlwrap_iternext, PyPtr, (PyPtr,))
            end
        end
        iter_result = iterate(o)
        return pyjlwrap_new(jlWrapIteratorType, (o, Ref{Union{Nothing,typeof(iter_result)}}(iterate(o))))
    end
end

#########################################################################
# Broadcasting: if the object is iterable, return collect(o), and otherwise
#               return o.

@static if isdefined(Base.Broadcast, :broadcastable)
    function Base.Broadcast.broadcastable(o::PyObject)
        iter = ccall((@pysym :PyObject_GetIter), PyPtr, (PyPtr,), o)
        if iter == C_NULL
            pyerr_clear()
            return Ref(o)
        else
            ccall(@pysym(:Py_DecRef), Cvoid, (PyPtr,), iter)
            return collect(o)
        end
    end
end
