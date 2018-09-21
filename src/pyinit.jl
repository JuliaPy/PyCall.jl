# Initializing Python (surprisingly complicated; see also deps/build.jl)

#########################################################################

# global PyObject constants, initialized to NULL and then overwritten in __init__
# (eventually, the ability to define global const in __init__ may go away,
#  and in any case this is better for type inference during precompilation)
const inspect = PyNULL()
const builtin = PyNULL()
const BuiltinFunctionType = PyNULL()
const MethodType = PyNULL()
const MethodWrapperType = PyNULL()
const ufuncType = PyNULL()
const format_traceback = PyNULL()
const pyproperty = PyNULL()
const jlfun2pyfun = PyNULL()
const c_void_p_Type = PyNULL()

# other global constants initialized at runtime are defined via Ref
# or are simply left as non-const values
const pynothing = Ref{PyPtr}(0)
const pyxrange = Ref{PyPtr}(0)

#########################################################################
# initialize jlWrapType for pytype.jl

function pyjlwrap_init()
    # PyMemberDef stores explicit pointers, hence must be initialized at runtime
    push!(pyjlwrap_members, PyMemberDef(pyjlwrap_membername,
                                        T_PYSSIZET, sizeof_PyObject_HEAD, READONLY,
                                        pyjlwrap_doc),
                            PyMemberDef(C_NULL,0,0,0,C_NULL))

    # all cfunctions must be compiled at runtime
    pyjlwrap_dealloc_ptr = @cfunction(pyjlwrap_dealloc, Cvoid, (PyPtr,))
    pyjlwrap_repr_ptr = @cfunction(pyjlwrap_repr, PyPtr, (PyPtr,))
    pyjlwrap_hash_ptr = @cfunction(pyjlwrap_hash, UInt, (PyPtr,))
    pyjlwrap_hash32_ptr = @cfunction(pyjlwrap_hash32, UInt32, (PyPtr,))
    pyjlwrap_call_ptr = @cfunction(pyjlwrap_call, PyPtr, (PyPtr,PyPtr,PyPtr))
    pyjlwrap_getattr_ptr = @cfunction(pyjlwrap_getattr, PyPtr, (PyPtr,PyPtr))
    pyjlwrap_getiter_ptr = @cfunction(pyjlwrap_getiter, PyPtr, (PyPtr,))

    # detect at runtime whether we are using Stackless Python
    try
        pyimport("stackless")
        Py_TPFLAGS_HAVE_STACKLESS_EXTENSION[] = Py_TPFLAGS_HAVE_STACKLESS_EXTENSION_
    catch
    end

    PyTypeObject!(jlWrapType, "PyCall.jlwrap", sizeof(Py_jlWrap)) do t::PyTypeObject
        t.tp_flags |= Py_TPFLAGS_BASETYPE
        t.tp_members = pointer(pyjlwrap_members);
        t.tp_dealloc = pyjlwrap_dealloc_ptr
        t.tp_repr = pyjlwrap_repr_ptr
        t.tp_call = pyjlwrap_call_ptr
        t.tp_getattro = pyjlwrap_getattr_ptr
        t.tp_iter = pyjlwrap_getiter_ptr
        t.tp_hash = sizeof(Py_hash_t) < sizeof(Int) ?
                    pyjlwrap_hash32_ptr : pyjlwrap_hash_ptr
    end
end

#########################################################################
# Virtual environment support

_clength(x::Cstring) = ccall(:strlen, Csize_t, (Cstring,), x) + 1
_clength(x) = length(x)

function __leak(::Type{T}, x) where T
    n = _clength(x)
    ptr = ccall(:malloc, Ptr{T}, (Csize_t,), n * sizeof(T))
    unsafe_copyto!(ptr, pointer(x), n)
    return ptr
end

"""
    _leak(T::Type, x::AbstractString) :: Ptr
    _leak(x::Array) :: Ptr

Leak `x` from Julia's GCer.  This is meant to be used only for
`Py_SetPythonHome` and `Py_SetProgramName` where the Python
documentation demands that the passed argument must points to "static
storage whose contents will not change for the duration of the
program's execution" (although it seems that in newer CPython versions
the contents are copied internally).
"""
_leak(x::Union{Cstring, Array}) = __leak(eltype(x), x)
_leak(T::Type, x::AbstractString) =
    _leak(Base.unsafe_convert(T, Base.cconvert(T, x)))
_leak(::Type{Cwstring}, x::AbstractString) =
    _leak(Base.cconvert(Cwstring, x))

function pythonhome_of(pyprogramname::AbstractString)
    if Sys.iswindows()
        script = """
        import sys
        sys.stdout.write(sys.exec_prefix)
        """
        # See where PYTHONHOME is mentioned in ../deps/build.jl
    else
        script = """
        import sys
        sys.stdout.write(sys.prefix)
        sys.stdout.write(":")
        sys.stdout.write(sys.exec_prefix)
        """
        # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHOME
    end
    cmd = `$pyprogramname -c $script`

    # For Windows:
    env = copy(ENV)
    env["PYTHONIOENCODING"] = "UTF-8"
    cmd = setenv(cmd, env)

    return read(cmd, String)
end

function find_libpython(python::AbstractString)
    script = joinpath(@__DIR__, "..", "deps", "find_libpython.py")
    try
        return read(`$python $script`, String)
    catch
        return nothing
    end
end

#########################################################################

function __init__()
    # sanity check: in Pkg for Julia 0.7+, the location of Conda can change
    # if e.g. you checkout Conda master, and we'll need to re-build PyCall
    # for something like pyimport_conda to continue working.
    if conda && dirname(python) != abspath(Conda.PYTHONDIR)
        error("Using Conda.jl python, but location of $python seems to have moved to $(Conda.PYTHONDIR).  Re-run Pkg.build(\"PyCall\") and restart Julia.")
    end

    # issue #189
    libpy_handle = libpython === nothing ? C_NULL :
        Libdl.dlopen(libpython, Libdl.RTLD_LAZY|Libdl.RTLD_DEEPBIND|Libdl.RTLD_GLOBAL)

    already_inited = 0 != ccall((@pysym :Py_IsInitialized), Cint, ())

    if already_inited
        # Importing from PyJulia takes this path.
    elseif isfile(get(ENV, "PYCALL_JL_RUNTIME_PYTHON", ""))
        venv_python = ENV["PYCALL_JL_RUNTIME_PYTHON"]

        # Check libpython compatibility.
        venv_libpython = find_libpython(venv_python)
        if venv_libpython === nothing
            error("""
            `libpython` for $venv_python cannot be found.
            PyCall.jl cannot initialize Python safely.
            """)
        elseif venv_libpython != libpython
            error("""
            Incompatible `libpython` detected.
            `libpython` for $venv_python is:
                $venv_libpython
            `libpython` for $pyprogramname is:
                $libpython
            PyCall.jl only supports loading Python environment using
            the same `libpython`.
            """)
        end

        if haskey(ENV, "PYCALL_JL_RUNTIME_PYTHONHOME")
            venv_home = ENV["PYCALL_JL_RUNTIME_PYTHONHOME"]
        else
            venv_home = pythonhome_of(venv_python)
        end
        if pyversion.major < 3
            ccall((@pysym :Py_SetPythonHome), Cvoid, (Cstring,),
                  _leak(Cstring, venv_home))
            ccall((@pysym :Py_SetProgramName), Cvoid, (Cstring,),
                  _leak(Cstring, venv_python))
        else
            ccall((@pysym :Py_SetPythonHome), Cvoid, (Ptr{Cwchar_t},),
                  _leak(Cwstring, venv_home))
            ccall((@pysym :Py_SetProgramName), Cvoid, (Ptr{Cwchar_t},),
                  _leak(Cwstring, venv_python))
        end
        ccall((@pysym :Py_InitializeEx), Cvoid, (Cint,), 0)
    else
        Py_SetPythonHome(libpy_handle, PYTHONHOME, wPYTHONHOME, pyversion)
        if !isempty(pyprogramname)
            if pyversion.major < 3
                ccall((@pysym :Py_SetProgramName), Cvoid, (Cstring,), pyprogramname)
            else
                ccall((@pysym :Py_SetProgramName), Cvoid, (Ptr{Cwchar_t},), wpyprogramname)
            end
        end
        ccall((@pysym :Py_InitializeEx), Cvoid, (Cint,), 0)
    end

    # Will get reinitialized properly on first use
    Compat.Sys.iswindows() && (PyActCtx[] = C_NULL)

    # Make sure python wasn't upgraded underneath us
    new_pyversion = vparse(split(unsafe_string(ccall(@pysym(:Py_GetVersion),
                               Ptr{UInt8}, ())))[1])

    if new_pyversion.major != pyversion.major
        error("PyCall precompiled with Python $pyversion, but now using Python $new_pyversion; ",
              "you need to relaunch Julia and re-run Pkg.build(\"PyCall\")")
    end

    copy!(inspect, pyimport("inspect"))
    copy!(builtin, pyimport(pyversion.major < 3 ? "__builtin__" : "builtins"))
    copy!(pyproperty, pybuiltin(:property))

    pyexc_initialize() # mappings from Julia Exception types to Python exceptions

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

    # jlwrap is a class, and when assigning it to an object
    #    obj[:foo] = some_julia_function
    # it won't behave like a regular Python method because it's not a Python
    # function (in particular, `self` won't be passed to it). The solution is:
    #    obj[:foo] = jlfun2pyfun(some_julia_function)
    # This is a bit of a kludge, obviously.
    copy!(jlfun2pyfun,
          pyeval_("""lambda f: lambda *args, **kwargs: f(*args, **kwargs)"""))

    if !already_inited
        # some modules (e.g. IPython) expect sys.argv to be set
        @static if VERSION >= v"0.7.0-DEV.1963"
            ref0 = Ref{UInt32}(0)
            GC.@preserve ref0 ccall(@pysym(:PySys_SetArgvEx), Cvoid,
                                         (Cint, Ref{Ptr{Cvoid}}, Cint),
                                         1, pointer_from_objref(ref0), 0)
        else
            ccall(@pysym(:PySys_SetArgvEx), Cvoid, (Cint, Ptr{Cwstring}, Cint),
                  1, [""], 0)
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
