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

# To pass an arbitrary Julia object to Python, we wrap it
# in a jlwrap Python class, where jlwrap.__call__
# executes the pyjlwrap_call function in Julia, which
# in turn fetches the actual Julia value from the python wrapper
# and calls it.

function pyjlwrap_call(self_::PyPtr, args_::PyPtr, kw_::PyPtr)
    args = PyObject(args_) # don't need pyincref because of finally clause below
    try
        f = unsafe_pyjlwrap_to_objref(self_)

        # on 0.6 we need to use invokelatest to get execution in newest world
        @static if isdefined(Base, :invokelatest)
            if kw_ == C_NULL
                ret = PyObject(Base.invokelatest(f, convert(PyAny, args)...))
            else
                kw = PyDict{Symbol,PyAny}(pyincref(kw_))
                kwargs = [ (k,v) for (k,v) in kw ]

                # 0.6 `invokelatest` doesn't support kwargs, instead
                # use a closure over kwargs. see:
                #   https://github.com/JuliaLang/julia/pull/22646
                f_kw_closure() = f(convert(PyAny, args)...; kwargs...)
                ret = PyObject(Core._apply_latest(f_kw_closure))
            end
        else # 0.5 support
            if kw_ == C_NULL
                ret = PyObject(f(convert(PyAny, args)...))
            else
                kw = PyDict{Symbol,PyAny}(pyincref(kw_))
                kwargs = [ (k,v) for (k,v) in kw ]
                ret = PyObject(f(convert(PyAny, args)...; kwargs...))
            end
        end

        return pystealref!(ret)
    catch e
        pyraise(e)
    finally
        args.o = PyPtr_NULL # don't decref
    end
    return PyPtr_NULL
end

pycallback(f::Function) = PyObject(f)