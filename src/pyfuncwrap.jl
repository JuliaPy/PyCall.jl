struct PyFuncWrap{P<:Union{PyObject,PyPtr}, AT<:Tuple, N, RT}
    o::P
    oargs::Vector{PyObject}
    pyargsptr::PyPtr
    ret::PyObject
end

"""
```
PyFuncWrap(o::P, argtypes::Tuple #= of Types =#, returntype::Type)
```

Wrap a callable PyObject/PyPtr to reduce the number of allocations made for
passing its arguments, and its return value, sometimes providing a speedup.
Mainly useful for functions called in a tight loop, particularly if most or
all of the arguments to the function don't change.
```
@pyimport numpy as np
rand22fn = PyFuncWrap(np.random["rand"], (Int, Int), PyArray)
setargs!(rand22fn, 2, 2)
for i in 1:10^9
    arr = rand22fn()
    ...
end
```
"""
function PyFuncWrap(o::P, argtypes::Tuple{Vararg{<:Union{Tuple, Type}}},
            returntype::Type{RT}=PyObject) where {P<:Union{PyObject,PyPtr}, RT}
    AT = typeof(argtypes)
    isvatuple(AT) && throw(ArgumentError("Vararg functions not supported, arg signature provided: $AT"))
    N = tuplen(AT)
    oargs = Array{PyObject}(N)
    pyargsptr = ccall((@pysym :PyTuple_New), PyPtr, (Int,), N)
    return PyFuncWrap{P, AT, N, RT}(o, oargs, pyargsptr, PyNULL())
end

"""
```
setargs!(pf::PyFuncWrap, args...)
```
Set the arguments to a python function wrapped in a PyFuncWrap, and convert them
to `PyObject`s that can be passed directly to python when the function is
called. After the arguments have been set, the function can be efficiently
called with `pf()`
"""
function setargs!(pf::PyFuncWrap{P, AT, N, RT}, args...) where {P, AT, RT, N}
    for i = 1:N
        setarg!(pf, args[i], i)
    end
    nothing
end

"""
```
setarg!(pf::PyFuncWrap, arg, i::Integer=1)
```
Set the `i`th argument to a python function wrapped in a PyFuncWrap, and convert
it to a `PyObject` that can be passed directly to python when the function is
called. Useful if a function takes multiple arguments, but only one or two of
them change, when calling the function in a tight loop
"""
function setarg!(pf::PyFuncWrap{P, AT, N, RT}, arg, i::Integer=1) where {P, AT, N, RT}
    pf.oargs[i] = PyObject(arg)
    @pycheckz ccall((@pysym :PyTuple_SetItem), Cint,
                     (PyPtr,Int,PyPtr), pf.pyargsptr, i-1, pf.oargs[i])
    pyincref(pf.oargs[i]) # PyTuple_SetItem steals the reference
    nothing
end

function (pf::PyFuncWrap{P, AT, N, RT})(args...) where {P, AT, N, RT}
    setargs!(pf, args...)
    return pf()
end

"""
Warning: if pf(args) or setargs(pf, ...) hasn't been called yet, this will likely segfault
"""
function (pf::PyFuncWrap{P, AT, N, RT})() where {P, AT, N, RT}
    sigatomic_begin()
    try
        kw = C_NULL
        retptr = ccall((@pysym :PyObject_Call), PyPtr, (PyPtr,PyPtr,PyPtr), pf.o,
                        pf.pyargsptr, kw)
        pyincref_(retptr)
        pf.ret.o = retptr
    finally
        sigatomic_end()
    end
    convert(RT, pf.ret)
end
