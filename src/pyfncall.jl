
# pyarg_tuples[i] is a pointer to a python tuple of length i-1
# const pyarg_tuples = Vector{PyPtr}(32)
const pyarg_tuples = PyPtr[]

"""
Low-level version of `pycall!(ret, o, ...; kwargs...)`
Sets `ret.o` to the result of the call, and returns `ret::PyObject`
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
Low-level version of `pycall!(ret, o, ...)` for when `kw` is already in python
friendly format but you don't have the python tuple to hold the arguments
(`pyargsptr`). Sets `ret.o` to the result of the call, and returns `ret::PyObject`.
"""
function _pycall!(ret::PyObject, o::Union{PyObject,PyPtr}, args,
  nargs::Int=length(args), kw::Union{Ptr{Void}, PyObject}=C_NULL)
    # pyarg_tuples[i] is a pointer to a python tuple of length i-1
    for n in length(pyarg_tuples):nargs
        push!(pyarg_tuples, @pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), n))
    end
    check_pyargsptr(nargs)
    pyargsptr = pyarg_tuples[nargs+1]
    return _pycall!(ret, pyargsptr, o, args, nargs, kw) #::PyObject
end

"""
Handle the situation where a callee had previously called incref on the
arguments tuple that was passed to it. We need to hold the only reference to the
arguments tuple, since setting a tuple item is only allowed when there is just
one reference to the tuple (tuples are supposed to be immutable in Python in all
other cases). Note that this actually happens when creating new builtin
exceptions, ref: https://github.com/python/cpython/blob/480ab05d5fee2b8fa161f799af33086a4e68c7dd/Objects/exceptions.c#L48
OTOH this py"def foo(*args): global z; z=args" doesn't trigger this.
Fortunately, this check for ob_refcnt is fast - only a few cpu clock cycles.
"""
function check_pyargsptr(nargs::Int)
    if nargs > 0 && unsafe_load(pyarg_tuples[nargs+1]).ob_refcnt > 1
        pydecref_(pyarg_tuples[nargs+1])
        pyarg_tuples[nargs+1] =
            @pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), nargs)
    end
end

"""
Low-level version of `pycall!(ret, o, ...)` for when `kw` is already in python
friendly format and you have the python tuple to hold the arguments (`pyargsptr`).
Sets the tuple's values to the python version of your arguments, and calls the
function. Sets `ret.o` to the result of the call, and returns `ret::PyObject`.
"""
function _pycall!(ret::PyObject, pyargsptr::PyPtr, o::Union{PyObject,PyPtr},
  args, nargs::Int=length(args), kw::Union{Ptr{Void}, PyObject}=C_NULL)
    pysetargs!(pyargsptr, args, nargs)
    return __pycall!(ret, pyargsptr, o, kw) #::PyObject
end

"""
```
pysetargs!(pyargsptr::PyPtr, args...)
```
Convert `args` to `PyObject`s, and set them as the elements of the Python tuple
pointed to by `pyargsptr`
"""
function pysetargs!(pyargsptr::PyPtr, args, N::Int)
    for i = 1:N
        pysetarg!(pyargsptr, args[i], i)
    end
end

"""
```
pysetarg!(pyargsptr::PyPtr, arg, i::Integer=1)
```
Convert `arg` to a `PyObject`, and set it as the `i-1`th element of the Python
tuple pointed to by `pyargsptr`
"""
function pysetarg!(pyargsptr::PyPtr, arg, i::Integer=1)
    pyarg = PyObject(arg)
    pyincref(pyarg) # PyTuple_SetItem steals the reference
    @pycheckz ccall((@pysym :PyTuple_SetItem), Cint,
                     (PyPtr,Int,PyPtr), pyargsptr, i-1, pyarg)
end

"""
Lowest level version of  `pycall!(ret, o, ...)`, assumes `pyargsptr` and `kw`
have all their args set to Python values, so we can just call the function `o`.
Sets `ret.o` to the result of the call, and returns `ret::PyObject`.
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

"""
```
pycall!(ret::PyObject, o::Union{PyObject,PyPtr}, returntype::Type, args...; kwargs...)
```
Set `ret` to the result of `pycall(o, returntype, args...; kwargs)` and return
`convert(returntype, ret)`.
Avoids allocating an extra PyObject for `ret`. See `pycall` for other details.
"""
pycall!(ret::PyObject, o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...) =
    return convert(returntype, _pycall!(ret, o, args, kwargs))

function pycall!(ret::PyObject, o::T, returntype::Type{PyObject}, args...; kwargs...) where
  T<:Union{PyObject,PyPtr}
    return _pycall!(ret, o, args, kwargs)
end

pycall!(ret::PyObject, o::Union{PyObject,PyPtr}, ::Type{PyAny}, args...; kwargs...) =
    return convert(PyAny, _pycall!(ret, o, args, kwargs))

"""
```
pycall(o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...)
```
Call the given Python function (typically looked up from a module) with the
given args... (of standard Julia types which are converted automatically to the
corresponding Python types if possible), converting the return value to
returntype (use a returntype of PyObject to return the unconverted Python object
reference, or of PyAny to request an automated conversion)
"""
pycall(o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...) =
    return convert(returntype, _pycall!(PyNULL(), o, args, kwargs))::returntype

pycall(o::Union{PyObject,PyPtr}, ::Type{PyAny}, args...; kwargs...) =
    return convert(PyAny, _pycall!(PyNULL(), o, args, kwargs))

(o::PyObject)(args...; kwargs...) =
    return convert(PyAny, _pycall!(PyNULL(), o, args, kwargs))

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
