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

# Static buffer to make sure the string passed to libpython persists
# for the lifetime of the program, as CPython API requires:
const __venv_programname = Vector{UInt8}(undef, 1024)
const __venv_pythonhome = Vector{UInt8}(undef, 1024)

"""
    _preserveas!(dest::Vector{UInt8}, (Cstring|Cwstring), x::String) :: Ptr

Copy `x` as `Cstring` or `Cwstring` to `dest`.
"""
function _preserveas!(dest::Vector{UInt8}, ::Type{Cstring}, x::AbstractString)
    s = transcode(UInt8, String(x))
    copyto!(dest, s)
    dest[length(s) + 1] = 0
    return pointer(dest)
end

function _preserveas!(dest::Vector{UInt8}, ::Type{Cwstring}, x::AbstractString)
    s = Base.cconvert(Cwstring, x)
    copyto!(reinterpret(Int32, dest), s)
    return pointer(dest)
end

venv_python(::Nothing) = pyprogramname

function venv_python(venv::AbstractString, suffix::AbstractString = "")
    # See:
    # https://github.com/python/cpython/blob/3.7/Lib/venv/__init__.py#L116
    if Compat.Sys.iswindows()
        return joinpath(venv, "Scripts", "python$suffix.exe")
    else
        return joinpath(venv, "bin", "python$suffix")
    end
end

"""
    python_cmd(args::Cmd = ``; venv, python) :: Cmd

Create an appropriate `Cmd` for running Python program with command
line arguments `args`.

# Keyword Arguments
- `venv::String`: The path of a virtualenv to be used instead of the
  default environment with which PyCall isconfigured.
- `python::String`: The path to the Python executable.  `venv` is ignored
  when this argument is specified.
"""
function python_cmd(args::Cmd = ``;
                    venv::Union{Nothing, AbstractString} = nothing,
                    python::AbstractString = venv_python(venv))
    return pythonenv(`$python $args`)
end

function find_libpython(python::AbstractString)
    script = joinpath(@__DIR__, "..", "deps", "find_libpython.py")
    cmd = python_cmd(`$script`; python = python)
    try
        return read(cmd, String)
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
        _current_python[] = ENV["PYCALL_JL_RUNTIME_PYTHON"]

        # Check libpython compatibility.
        venv_libpython = find_libpython(current_python())
        if venv_libpython === nothing
            error("""
            `libpython` for $(current_python()) cannot be found.
            PyCall.jl cannot initialize Python safely.
            """)
        elseif venv_libpython != libpython
            error("""
            Incompatible `libpython` detected.
            `libpython` for $(current_python()) is:
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
            venv_home = pythonhome_of(current_python())
        end
        if pyversion.major < 3
            ccall((@pysym :Py_SetPythonHome), Cvoid, (Cstring,),
                  _preserveas!(__venv_pythonhome, Cstring, venv_home))
            ccall((@pysym :Py_SetProgramName), Cvoid, (Cstring,),
                  _preserveas!(__venv_programname, Cstring, current_python()))
        else
            ccall((@pysym :Py_SetPythonHome), Cvoid, (Ptr{Cwchar_t},),
                  _preserveas!(__venv_pythonhome, Cwstring, venv_home))
            ccall((@pysym :Py_SetProgramName), Cvoid, (Ptr{Cwchar_t},),
                  _preserveas!(__venv_programname, Cwstring, current_python()))
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
