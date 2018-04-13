
# pyarg_tuples[i] is a pointer to a python tuple of length i-1
# const pyarg_tuples = Vector{PyPtr}(32)
const pyarg_tuples = PyPtr[]
# const cur_args_size = Ref{Int}(0)
"""
You have the args, but kwargs need to be converted to a Ptr to a Python dict, or
C_NULL if empty
"""
function _pycall!(ret::PyObject, o::Union{PyObject,PyPtr}, args, kwargs::Vector{Any})
    if isempty(kwargs)
        kw = C_NULL
    else
        kw = PyObject(Dict{AbstractString, Any}([Pair(string(k), v) for (k, v) in kwargs]))
    end
    _pycall!(ret, o, args, length(args), kw)
end

"""
You have the args, and kwargs is in python friendly format, but
you don't yet have the python tuple to hold the arguments.
"""
function _pycall!(ret::PyObject, o::Union{PyObject,PyPtr}, args,
  nargs::Int=length(args), kw::Union{Ptr{Void}, PyObject}=C_NULL)
    # pyarg_tuples[i] is a pointer to a python tuple of length i-1
    for n in length(pyarg_tuples):nargs
        push!(pyarg_tuples, @pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), n))
    end
    pyargsptr = pyarg_tuples[nargs+1]
    return _pycall!(ret, pyargsptr, o, args, nargs, kw) #::PyObject
end

"""
You have the args, and kwargs is in python friendly format, and
you have the python tuple to hold the arguments (pyargsptr), you just haven't
set the tuples values to the python version of your arguments yet
"""
function _pycall!(ret::PyObject, pyargsptr::PyPtr, o::Union{PyObject,PyPtr},
  args, nargs::Int=length(args), kw::Union{Ptr{Void}, PyObject}=C_NULL)
    setargs!(pyargsptr, args, nargs)
    return __pycall!(ret, pyargsptr, o, kw) #::PyObject
end

"""
You're all frocked up and ready for the dance. `pyargsptr` and `kw` have all
their args set to Python values, so we can call the function.
"""
function __pycall!(ret::PyObject, pyargsptr::PyPtr, o::Union{PyObject,PyPtr},
  kw::Union{Ptr{Void}, PyObject})
    sigatomic_begin()
    try
        retptr = @pycheckn ccall((@pysym :PyObject_Call), PyPtr, (PyPtr,PyPtr,PyPtr), o,
                        pyargsptr, kw)
        pyincref_(retptr)
        pydecref(ret)
        ret.o = retptr
    finally
        sigatomic_end()
    end
    return ret #::PyObject
end

pycall!(pyobj::PyObject, o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...) =
    return convert(returntype, _pycall!(pyobj, o, args, kwargs))

function pycall!(pyobj::PyObject, o::T, returntype::Type{PyObject}, args...; kwargs...) where
  T<:Union{PyObject,PyPtr}
    return _pycall!(pyobj, o, args, kwargs)
end

pycall!(pyobj::PyObject, o::Union{PyObject,PyPtr}, ::Type{PyAny}, args...; kwargs...) =
    return convert(PyAny, _pycall!(pyobj, o, args, kwargs))

"""
    pycall(o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...)

Call the given Python function (typically looked up from a module) with the given args... (of standard Julia types which are converted automatically to the corresponding Python types if possible), converting the return value to returntype (use a returntype of PyObject to return the unconverted Python object reference, or of PyAny to request an automated conversion)
"""
pycall(o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...) =
    return convert(returntype, _pycall!(PyNULL(), o, args, kwargs))::returntype

pycall(o::Union{PyObject,PyPtr}, ::Type{PyAny}, args...; kwargs...) =
    return convert(PyAny, _pycall!(PyNULL(), o, args, kwargs))

(o::PyObject)(args...; kws...) = pycall(o, PyAny, args...; kws...)
PyAny(o::PyObject) = convert(PyAny, o)

"""
    @pycall func(args...)::T

Convenience macro which turns `func(args...)::T` into pycall(func, T, args...)
"""
macro pycall(ex)
    if !(isexpr(ex,:(::)) && isexpr(ex.args[1],:call))
        throw(ArgumentError("Usage: @pycall func(args...)::T"))
    end
    func = ex.args[1].args[1]
    args, kwargs = ex.args[1].args[2:end], []
    if isexpr(args[1],:parameters)
        kwargs, args = args[1], args[2:end]
    end
    T = ex.args[2]
    :(pycall($(map(esc,[kwargs; func; T; args])...)))
end

#########################################################################
"""
```
pywrapfn(o::PyObject, nargs::Int, returntype::Type{T}=PyObject) where T
```

Wrap a callable PyObject/PyPtr to reduce the number of allocations made for
passing its arguments, and its return value, sometimes providing a speedup.
Mainly useful for functions called in a tight loop. After wrapping, arguments
should be passed in tuple, rather than directly, e.g. `wrappedfn((a,b))` rather
than `wrappedfn(a,b)`
```
@pyimport numpy as np

# wrap a 2-arg version of np.random.rand for creating random matrices
randmatfn = pywrapfn(np.random["rand"], 2, PyArray)

# n.b. rand would normally take multiple arguments, like so:
a_random_matrix = np.random["rand"](7, 7)

# but we call the wrapped version with a tuple instead, i.e.
# rand22fn((4, 4)) not
# rand22fn(4, 4)
for i in 1:10^9
    arr = rand22fn((4, 4))
    ...
end
```
"""
function pywrapfn(o::PyObject, nargs::Int, returntype::Type{T}=PyObject) where T
    pyargsptr = ccall((@pysym :PyTuple_New), PyPtr, (Int,), nargs)
    ret = PyNULL()
    wrappedfn(args) = convert(T, _pycall!(ret, pyargsptr, o, args, nargs, C_NULL))
    wrappedfn() = convert(T, __pycall!(ret, pyargsptr, o, C_NULL))
    return wrappedfn
end

"""
```
setargs!(pyargsptr::PyPtr, args...)
```
Set the arguments to a python function, and convert them
to `PyObject`s that can be passed directly to python when the function is
called.
"""
function setargs!(pyargsptr::PyPtr, args::Tuple, N::Int)
    for i = 1:N
        setarg!(pyargsptr, args[i], i)
    end
    nothing
end

"""
```
setarg!(pyargsptr::PyPtr, arg, i::Integer=1)
```
Set the `i`th argument to a python function, and convert
it to a `PyObject` that can be passed directly to python when the function is
called. Can be used if a function takes multiple arguments, but only one or two
of them change, when the function is called in a tight loop.
"""
function setarg!(pyargsptr::PyPtr, arg, i::Integer=1)
    pyarg = PyObject(arg)
    @pycheckz ccall((@pysym :PyTuple_SetItem), Cint,
                     (PyPtr,Int,PyPtr), pyargsptr, i-1, pyarg)
    pyincref(pyarg) # PyTuple_SetItem steals the reference
end
