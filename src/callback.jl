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
    @eval const $def = PyMethodDef($name, $f, $flags)
    PyObject(@pycheckn ccall((@pysym :PyCFunction_NewEx), PyPtr,
                             (Ptr{PyMethodDef}, Ptr{Void}, Ptr{Void}),
                             &eval(def), C_NULL, C_NULL))
end

################################################################

# To pass an arbitrary Julia function to Python, we wrap it
# in a jl_Function Python class, where jl_Function.__call__
# executes the jl_Function_callback function in Julia, which
# in turn fetches the actual Julia function from the "f" attribute
# and calls it.

function jl_Function_call(self_::PyPtr, args_::PyPtr, kw_::PyPtr)
    ret_ = convert(PyPtr, C_NULL)
    args = PyObject(args_)
    try
        f = unsafe_pyjlwrap_to_objref(self_)::Function
        if kw_ == C_NULL
            ret = PyObject(f(convert(PyAny, args)...))
        else
            kw = PyDict{Symbol,PyAny}(PyObject(kw_))
            kwargs = [ (k,v) for (k,v) in kw ]
            ret = PyObject(f(convert(PyAny, args)...; kwargs...))
        end
        ret_ = ret.o
        ret.o = convert(PyPtr, C_NULL) # don't decref
    catch e
        pyraise(e)
    finally
        args.o = convert(PyPtr, C_NULL) # don't decref
    end
    return ret_::PyPtr
end
const jl_Function_call_ptr = cfunction(jl_Function_call,
                                       PyPtr, (PyPtr,PyPtr,PyPtr))

jl_FunctionType = PyTypeObject()

function pycallback_initialize()
    global jl_FunctionType
    if (jl_FunctionType::PyTypeObject).tp_name == C_NULL
        jl_FunctionType::PyTypeObject = 
         pyjlwrap_type("PyCall.jl_Function",
                       t -> t.tp_call = jl_Function_call_ptr)
    end
    return
end

function pycallback_finalize()
    global jl_FunctionType
    jl_FunctionType::PyTypeObject = PyTypeObject()
end

function pycallback(f::Function) 
    global jl_FunctionType
    if (jl_FunctionType::PyTypeObject).tp_name == C_NULL
        pycallback_initialize()
    end
    pyjlwrap_new(jl_FunctionType::PyTypeObject, f)
end

PyObject(f::Function) = pycallback(f)
