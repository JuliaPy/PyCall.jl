# Initializing Python (surprisingly complicated; see also deps/build.jl)

#########################################################################

# global PyObject constants, initialized to NULL and then overwritten in __init__
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
const c_void_p_Type = PyNULL()

# other global constants initialized at runtime are defined via Ref
# or are simply left as non-const values
pyversion = pyversion_build # not a Ref since pyversion is exported
const pynothing = Ref{PyPtr}()
const pyxrange = Ref{PyPtr}()

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

    # Will get reinitialized properly on first use
    is_windows() && (PyActCtx[] = C_NULL)

    # cache the Python version as a Julia VersionNumber
    global pyversion = convert(VersionNumber, split(unsafe_string(ccall(@pysym(:Py_GetVersion),
                               Ptr{UInt8}, ())))[1])
    if pyversion_build.major != pyversion.major
        error("PyCall built with Python $pyversion_build, but now using Python $pyversion; ",
              "you need to relaunch Julia and re-run Pkg.build(\"PyCall\")")
    end

    copy!(inspect, pyimport("inspect"))
    copy!(builtin, pyimport(pyversion_build.major < 3 ? "__builtin__" : "builtins"))
    copy!(pyproperty, pybuiltin(:property))

    pyexc_initialize() # mappings from Julia Exception types to Python exceptions

    types = pyimport("types")
    copy!(TypeType, pybuiltin("type")) # for pytypeof

    # cache Python None -- PyPtr, not PyObject, to prevent it from
    # being finalized prematurely on exit
    pynothing[] = @pyglobalobj(:_Py_NoneStruct)

    # xrange type (or range in Python 3)
    pyxrange[] = @pyglobalobj(:PyRange_Type)

    # ctypes.c_void_p for Ptr types
    copy!(c_void_p_Type, pyimport("ctypes")["c_void_p"])

    # traceback.format_tb function, for show(PyError)
    copy!(format_traceback, pyimport("traceback")["format_tb"])

    init_datetime()
    pyjlwrap_init()

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
        if pyversion_build.major < 3
            argv_s = Compat.String("")
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
