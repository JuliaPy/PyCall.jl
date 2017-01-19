# Passing Julia callback functions to Python routines.
#
# Note that this will typically involve two functions: the
# desired Julia function/closure, and a top-level C-callable
# wrapper function used with PyCFunction_NewEx -- the latter
# is called from Python and calls the former as needed.

################################################################

# Define a Python method/function object from f(PyPtr,PyPtr)::PyPtr.
# Requires f to be a top-level function.
function pymethod(f::Function, name::AbstractString, flags::Integer)
    # Python expects the PyMethodDef structure to be a *constant*,
    # so we define an anonymous global to hold it.
    def = gensym("PyMethodDef")
    @eval const $def = PyMethodDef[PyMethodDef($name, $f, $flags)]
    PyObject(@pycheckn ccall((@pysym :PyCFunction_NewEx), PyPtr,
                             (Ptr{PyMethodDef}, Ptr{Void}, Ptr{Void}),
                             eval(def), C_NULL, C_NULL))
end

################################################################

# To pass an arbitrary Julia function to Python, we wrap it
# in a jl_Function Python class, where jl_Function.__call__
# executes the jl_Function_callback function in Julia, which
# in turn fetches the actual Julia function from the "f" attribute
# and calls it.

function jl_Function_call(self_::PyPtr, args_::PyPtr, kw_::PyPtr)
    args = PyObject(args_) # don't need pyincref because of finally clause below
    try
        f = unsafe_pyjlwrap_to_objref(self_)::Function
        if kw_ == C_NULL
            ret = PyObject(f(convert(PyAny, args)...))
        else
            kw = PyDict{Symbol,PyAny}(pyincref(kw_))
            kwargs = [ (k,v) for (k,v) in kw ]
            ret = PyObject(f(convert(PyAny, args)...; kwargs...))
        end
        return pystealref!(ret)
    catch e
        pyraise(e)
    finally
        args.o = PyPtr_NULL # don't decref
    end
    return PyPtr_NULL
end

# just string(Docs.doc(f)) doesn't work on VERSION < v"0.5"
function docstring(f::Function)
    buf = IOBuffer()
    show(buf, "text/plain", Docs.doc(f))
    return String(take!(buf))
end

# this function emulates standard attributes of Python functions,
# where possible.
function jl_Function_getattr(self_::PyPtr, attr__::PyPtr)
    attr_ = PyObject(attr__) # don't need pyincref because of finally clause below
    try
        f = unsafe_pyjlwrap_to_objref(self_)::Function
        attr = convert(String, attr_)
        if attr in ("__name__","func_name")
            return pystealref!(PyObject(string(f)))
        elseif attr in ("__doc__", "func_doc")
            return pystealref!(PyObject(docstring(f)))
        elseif attr in ("__module__","__defaults__","func_defaults","__closure__","func_closure")
            return pystealref!(PyObject(nothing))
        else
            # TODO: handle __code__/func_code (issue #268)

            # fallback: (probably raises AttributeError)
            return ccall(@pysym(:PyObject_GenericGetAttr), PyPtr, (PyPtr,PyPtr), self_, attr__)
        end
    catch e
        pyraise(e)
    finally
        attr_.o = PyPtr_NULL # don't decref
    end
    return PyPtr_NULL
end

function pycallback(f::Function)
    pyjlwrap_new(jl_FunctionType, f)
end

PyObject(f::Function) = pycallback(f)
