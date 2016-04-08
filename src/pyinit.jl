# Initializing Python (surprisingly complicated; see also deps/build.jl)

#########################################################################
# global constants, initialized to NULL and then overwritten in __init__
# (eventually, the ability to define global const in __init__ may go away,
#  and in any case this is better for type inference during precompilation)

const inspect = PyNULL()
const builtin = PyNULL()
const BuiltinFunctionType = PyNULL()
const TypeType = PyNULL()
const MethodType = PyNULL()
const MethodWrapperType = PyNULL()
const ufuncType = PyNULL()
const format_traceback = PyNULL()
const pyproperty = PyNULL()
const jlfun2pyfun = PyNULL()

#########################################################################

function __init__()
    # issue #189
    Libdl.dlopen(libpython, Libdl.RTLD_LAZY|Libdl.RTLD_DEEPBIND|Libdl.RTLD_GLOBAL)

    already_inited = 0 != ccall((@pysym :Py_IsInitialized), Cint, ())

    if !already_inited
        if !isempty(PYTHONHOME)
            if pyversion_build.major < 3
                ccall((@pysym :Py_SetPythonHome), Void, (Cstring,), PYTHONHOME)
            else
                ccall((@pysym :Py_SetPythonHome), Void, (Cwstring,), PYTHONHOME)
            end
        end
        if !isempty(pyprogramname)
            if pyversion_build.major < 3
                ccall((@pysym :Py_SetProgramName), Void, (Cstring,), pyprogramname)
            else
                ccall((@pysym :Py_SetProgramName), Void, (Cwstring,), pyprogramname)
            end
        end
        ccall((@pysym :Py_InitializeEx), Void, (Cint,), 0)
    end

    # cache the Python version as a Julia VersionNumber
    global const pyversion = convert(VersionNumber,
                                     split(bytestring(ccall(@pysym(:Py_GetVersion),
                                                            Ptr{UInt8}, ())))[1])
    if pyversion_build.major != pyversion.major
        error("PyCall built with Python $pyversion_build, but now using Python $pyversion; ",
              "you need to relaunch Julia and re-run Pkg.build(\"PyCall\")")
    end

    copy!(inspect, pyimport("inspect"))
    copy!(builtin, pyimport(pyversion.major < 3 ? "__builtin__" : "builtins"))
    copy!(pyproperty, pybuiltin(:property))

    pyexc_initialize() # mappings from Julia Exception types to Python exceptions

    types = pyimport("types")
    copy!(TypeType, pybuiltin("type")) # for pytypeof

    # cache Python None -- PyPtr, not PyObject, to prevent it from
    # being finalized prematurely on exit
    global const pynothing = @pyglobalobj(:_Py_NoneStruct)

    # xrange type (or range in Python 3)
    global const pyxrange = @pyglobalobj(:PyRange_Type)

    # cache ctypes.c_void_p type and function if available
    vpt, pvp = try
        (pyimport("ctypes")["c_void_p"],
         p::Ptr -> pycall(c_void_p_Type, PyObject, UInt(p)))
    catch # fallback to CObject
        (@pyglobalobj(:PyCObject_FromVoidPtr),
         p::Ptr -> PyObject(ccall(pycobject_new, PyPtr, (Ptr{Void}, Ptr{Void}), p, C_NULL)))
    end
    global const c_void_p_Type = vpt
    global const py_void_p = pvp

    # traceback.format_tb function, for show(PyError)
    copy!(format_traceback, pyimport("traceback")["format_tb"])

    # all cfunctions must be compiled at runtime
    global const jl_Function_call_ptr =
        cfunction(jl_Function_call, PyPtr, (PyPtr,PyPtr,PyPtr))
    global const pyjlwrap_dealloc_ptr = cfunction(pyjlwrap_dealloc, Void, (PyPtr,))
    global const pyjlwrap_repr_ptr = cfunction(pyjlwrap_repr, PyPtr, (PyPtr,))
    global const pyjlwrap_hash_ptr = cfunction(pyjlwrap_hash, UInt, (PyPtr,))
    global const pyjlwrap_hash32_ptr = cfunction(pyjlwrap_hash32, UInt32, (PyPtr,))

    # PyMemberDef stores explicit pointers, hence must be initialized in __init__
    global const pyjlwrap_members =
        PyMemberDef[ PyMemberDef(pyjlwrap_membername,
                                 T_PYSSIZET, sizeof_PyObject_HEAD, READONLY,
                                 pyjlwrap_doc),
                     PyMemberDef(C_NULL,0,0,0,C_NULL) ]

    init_datetime()
    pyjlwrap_init()

    global const jl_FunctionType = pyjlwrap_type("PyCall.jl_Function",
                                                 t -> t.tp_call =
                                                 jl_Function_call_ptr)

    # jl_FunctionType is a class, and when assigning it to an object
    #    obj[:foo] = some_julia_function
    # it won't behave like a regular Python method because it's not a Python
    # function (in particular, `self` won't be passed to it). The solution is:
    #    obj[:foo] = jlfun2pyfun(some_julia_function)
    # This is a bit of a kludge, obviously.
    copy!(jlfun2pyfun,
          pyeval("""lambda f: lambda *args, **kwargs: f(*args, **kwargs)"""))

    if !already_inited
        # some modules (e.g. IPython) expect sys.argv to be set
        if pyversion.major < 3
            argv_s = bytestring("")
            argv = unsafe_convert(Ptr{UInt8}, argv_s)
            ccall(@pysym(:PySys_SetArgvEx), Void, (Cint,Ptr{Ptr{UInt8}},Cint), 1, &argv, 0)
        else
            argv_s = Cwchar_t[0]
            argv   = unsafe_convert(Ptr{Cwchar_t}, argv_s)
            ccall(@pysym(:PySys_SetArgvEx), Void, (Cint, Ptr{Ptr{Cwchar_t}}, Cint), 1, &argv, 0)
        end

        # Some Python code checks sys.ps1 to see if it is running
        # interactively, and refuses to be interactive otherwise.
        # (e.g. Matplotlib: see PyPlot#79)
        if isinteractive()
            let sys = pyimport("sys")
                if !haskey(sys, "ps1")
                    sys["ps1"] = ">>> "
                end
            end
        end
    end
end
