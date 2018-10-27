using PyCall: @pycheckn, pyincref_, __pycall!

#########################################################################
struct PyWrapFn{N, RT}
    o::PyPtr
    pyargsptr::PyPtr
    ret::PyObject
end

function PyWrapFn(o::Union{PyObject, PyPtr}, nargs::Int, returntype::Type=PyObject)
    pyargsptr = ccall((@pysym :PyTuple_New), PyPtr, (Int,), nargs)
    return PyWrapFn{nargs, returntype}(pyincref_(PyPtr(o)), pyargsptr, PyNULL())
end

(pf::PyWrapFn{N, RT})(args...) where {N, RT} =
    convert(RT, _pycall!(pf.ret, pf.pyargsptr, pf.o, args, N, C_NULL))

(pf::PyWrapFn{N, RT})() where {N, RT} =
    convert(RT, __pycall!(pf.ret, pf.pyargsptr, pf.o, C_NULL))

"""
```
pywrapfn(o::PyObject, nargs::Int, returntype::Type{T}=PyObject) where T
```
Wrap a callable PyObject/PyPtr possibly making calling it more performant. The
wrapped version (of type `PyWrapFn`) reduces the number of allocations made for
passing its arguments, and re-uses the same PyObject as its return value each
time it is called.

Mainly useful for functions called in a tight loop. After wrapping, arguments
should be passed in a tuple, rather than directly, e.g. `wrappedfn((a,b))` rather
than `wrappedfn(a,b)`.
Example
```
@pyimport numpy as np

# wrap a 2-arg version of np.random.rand for creating random matrices
randmatfn = pywrapfn(np.random["rand"], 2, PyArray)

# n.b. rand would normally take multiple arguments, like so:
a_random_matrix = np.random["rand"](7, 7)

# but we call the wrapped version with a tuple instead, i.e.
# rand22fn((7, 7)) not
# rand22fn(7, 7)
for i in 1:10^9
    arr = rand22fn((7,7))
    ...
end
```
"""
pywrapfn(o::PyObject, nargs::Int, returntype::Type=PyObject) =
    PyWrapFn(o, nargs, returntype)

"""
```
pysetargs!(w::PyWrapFn{N, RT}, args)
```
Set the arguments with which to call a Python function wrapped using
`w = pywrapfn(pyfun, ...)`
"""
function pysetargs!(pf::PyWrapFn{N, RT}, args) where {N, RT}
    check_pyargsptr(pf)
    pysetargs!(pf.pyargsptr, args, N)
end

"""
```
pysetarg!(w::PyWrapFn{N, RT}, arg, i::Integer=1)
```
Set the `i`th argument to be passed to a Python function previously
wrapped with a call to `w = pywrapfn(pyfun, ...)`
"""
function pysetarg!(pf::PyWrapFn{N, RT}, arg, i::Integer=1) where {N, RT}
    check_pyargsptr(pf)
    pysetarg!(pf.pyargsptr, arg, i)
end

"""
See check_pyargsptr(nargs::Int) above
"""
function check_pyargsptr(pf::PyWrapFn{N, RT}) where {N, RT}
    if unsafe_load(pf.pyargsptr).ob_refcnt > 1
        pydecref_(pf.pyargsptr)
        pf.pyargsptr =
            @pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), nargs)
    end
end
