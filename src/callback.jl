# Passing Julia callback functions to Python routines.
#
# Note that this will typically involve two functions: the
# desired Julia function/closure, and a top-level C-callable
# wrapper function used with PyCFunction_NewEx -- the latter
# is called from Python and calls the former as needed.

################################################################

# Define a Python method/function object from f(PyPtr,PyPtr)::PyPtr.
# Requires f to be a top-level function.
function pymethod(f::Function, name::String, flags::Integer)
    # Python expects the PyMethodDef structure to be a *constant*,
    # and the strings pointed to therein must also be constants,
    # so we define anonymous globals to hold these
    def = gensym("PyMethodDef")
    defname = gensym("PyMethodDef_ml_name")
    @eval const $defname = bytestring($name)
    @eval const $def = PyMethodDef(convert(Ptr{Uint8}, $defname),
                                   $(cfunction(f, PyPtr, (PyPtr,PyPtr))),
                                   convert(Cint, $flags),
                                   convert(Ptr{Uint8}, C_NULL))
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
        if kw_ != C_NULL
            throw(ArgumentError("keywords not yet supported in callbacks"))
        end
        f = unsafe_pyjlwrap_to_objref(self_)::Function
        ret = PyObject(f(convert(PyAny, args)...))
        ret_ = ret.o
        ret.o = convert(PyPtr, C_NULL) # don't decref
    catch e
        ccall((@pysym :PyErr_SetString), Void, (PyPtr, Ptr{Uint8}),
              (@pysym :PyExc_RuntimeError),
              bytestring(string("Julia exception: ", e)))
    finally
        args.o = convert(PyPtr, C_NULL) # don't decref
    end
    return ret_::PyPtr
end

jl_FunctionType = PyTypeObject()

function pycallback_initialize()
    global jl_FunctionType
    if (jl_FunctionType::PyTypeObject).tp_name == C_NULL
        jl_FunctionType::PyTypeObject = 
         pyjlwrap_type("PyCall.jl_Function",
                       t -> t.tp_call = cfunction(jl_Function_call,
                                                  PyPtr, (PyPtr,PyPtr,PyPtr)))
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
