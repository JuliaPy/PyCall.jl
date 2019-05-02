# Support Python iterators in Julia and vice versa

#########################################################################
# Iterating over Python objects in Julia

Base.IteratorSize(::Type{PyObject}) = Base.SizeUnknown()
function _start(po::PyObject)
    disable_sigint() do
        o = PyObject(@pycheckn ccall((@pysym :PyObject_GetIter), PyPtr, (PyPtr,), po))
        nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), o))

        return (nxt,o)
    end
end

"""
    PyIterator{T}(pyobject)

Wrap `pyobject::PyObject` into an iterator, that produces items of type `T`. To be more precise `convert(T, item)` is applied in each iteration. This can be useful to avoid automatic conversion of items into corresponding julia types.
```jldoctest
julia> using PyCall

julia> l = PyObject([PyObject(1), PyObject(2)])
PyObject [1, 2]

julia> piter = PyCall.PyIterator{PyAny}(l)
PyCall.PyIterator{PyAny,Base.HasLength()}(PyObject [1, 2])

julia> collect(piter)
2-element Array{Any,1}:
    1
    2

julia> piter = PyCall.PyIterator(l)
PyCall.PyIterator{PyObject,Base.HasLength()}(PyObject [1, 2])

julia> collect(piter)
2-element Array{PyObject,1}:
    PyObject 1
    PyObject 2
```
"""
struct PyIterator{T,S}
    o::PyObject
end

function _compute_IteratorSize(o::PyObject)
    S = try
        length(o)
        Base.HasLength
    catch err
        if !(err isa PyError && pyisinstance(err.val, @pyglobalobjptr :PyExc_TypeError))
            rethrow()
        end
        Base.SizeUnknown
    end
end
function PyIterator(o::PyObject)
    PyIterator{PyObject}(o)
end
function (::Type{PyIterator{T}})(o::PyObject) where {T}
    S = _compute_IteratorSize(o)
    PyIterator{T,S}(o)
end

Base.eltype(::Type{<:PyIterator{T}}) where T = T
Base.eltype(::Type{<:PyIterator{PyAny}}) = Any
Base.length(piter::PyIterator) = length(piter.o)

Base.IteratorSize(::Type{<: PyIterator{T,S}}) where {T,S} = S()

_start(piter::PyIterator) = _start(piter.o)

function Base.iterate(piter::PyIterator{T}, s=_start(piter)) where {T}
    ispynull(s[1]) && return nothing
    disable_sigint() do
        nxt = PyObject(@pycheck ccall((@pysym :PyIter_Next), PyPtr, (PyPtr,), s[2]))
        return (convert(T,s[1]), (nxt, s[2]))
    end
end
function Base.iterate(po::PyObject, s=_start(po))
    # avoid the constructor that calls length
    # since that might be an expensive operation
    # even if length is cheap, this adds 10% performance
    piter = PyIterator{PyAny, Base.SizeUnknown}(po)
    iterate(piter, s)
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
        @pyraise e
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
        @pyraise e
    end
    return PyPtr_NULL
end

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

#########################################################################
# Broadcasting: if the object is iterable, return collect(o), and otherwise
#               return o.
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
