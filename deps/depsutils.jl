import Compat.Libdl

hassym(lib, sym) = Libdl.dlsym_e(lib, sym) != C_NULL

# call dlsym_e on a sequence of symbols and return the symbol that gives
# the first non-null result
function findsym(lib, syms...)
    for sym in syms
        if hassym(lib, sym)
            return sym
        end
    end
    error("no symbol found from: ", syms)
end

# Static buffer to make sure the string passed to libpython persists
# for the lifetime of the program, as CPython API requires:
const __buf_programname = Vector{UInt8}(undef, 1024)
const __buf_pythonhome = Vector{UInt8}(undef, 1024)

# Need to set PythonHome before calling GetVersion to avoid warning (#299).
# Unfortunately, this poses something of a chicken-and-egg problem because
# we need to know the Python version to set PythonHome via the API.  Note
# that the string (or array) passed to Py_SetPythonHome needs to be a
# constant that lasts for the lifetime of the program, which is why we
# prepare static buffer __buf_pythonhome, copy the string to it, and then
# pass the pointer to the buffer to the CPython API.
function Py_SetPythonHome(libpy, pyversion, PYTHONHOME::AbstractString)
    isempty(PYTHONHOME) && return
    if pyversion.major < 3
        ccall(Libdl.dlsym(libpy, :Py_SetPythonHome), Cvoid, (Cstring,),
              _preserveas!(__buf_pythonhome, Cstring, PYTHONHOME))
    else
        ccall(Libdl.dlsym(libpy, :Py_SetPythonHome), Cvoid, (Ptr{Cwchar_t},),
              _preserveas!(__buf_pythonhome, Cwstring, PYTHONHOME))
    end
end

function Py_SetProgramName(libpy, pyversion, programname::AbstractString)
    isempty(programname) && return
    if pyversion.major < 3
        ccall(Libdl.dlsym(libpy, :Py_SetProgramName), Cvoid, (Cstring,),
              _preserveas!(__buf_programname, Cstring, programname))
    else
        ccall(Libdl.dlsym(libpy, :Py_SetProgramName), Cvoid, (Ptr{Cwchar_t},),
              _preserveas!(__buf_programname, Cwstring, programname))
    end
end

"""
    _preserveas!(dest::Vector{UInt8}, (Cstring|Cwstring), x::String) :: Ptr

Copy `x` as `Cstring` or `Cwstring` to `dest` and return a pointer to
`dest`.  Thus, this pointer is safe to use as long as `dest` is
protected from GC.
"""
function _preserveas!(dest::Vector{UInt8}, ::Type{Cstring}, x::AbstractString)
    s = transcode(UInt8, String(x))
    copyto!(dest, s)
    dest[length(s) + 1] = 0
    return pointer(dest)
end

function _preserveas!(dest::Vector{UInt8}, ::Type{Cwstring}, x::AbstractString)
    s = Base.cconvert(Cwstring, x)
    copyto!(dest, reinterpret(UInt8, s))
    return pointer(dest)
end


# need to be able to get the version before Python is initialized
Py_GetVersion(libpy) = unsafe_string(ccall(Libdl.dlsym(libpy, :Py_GetVersion), Ptr{UInt8}, ()))

# Fix the environment for running `python`, and setts IO encoding to UTF-8.
# If cmd is the Conda python, then additionally removes all PYTHON* and
# CONDA* environment variables.
function pythonenv(cmd::Cmd)
    env = copy(ENV)
    if dirname(cmd.exec[1]) == abspath(Conda.PYTHONDIR)
        pythonvars = String[]
        for var in keys(env)
            if startswith(var, "CONDA") || startswith(var, "PYTHON")
                push!(pythonvars, var)
            end
        end
        for var in pythonvars
            pop!(env, var)
        end
    end
    # set PYTHONIOENCODING when running python executable, so that
    # we get UTF-8 encoded text as output (this is not the default on Windows).
    env["PYTHONIOENCODING"] = "UTF-8"
    setenv(cmd, env)
end


function pythonhome_of(pyprogramname::AbstractString)
    if Compat.Sys.iswindows()
        # PYTHONHOME tells python where to look for both pure python
        # and binary modules.  When it is set, it replaces both
        # `prefix` and `exec_prefix` and we thus need to set it to
        # both in case they differ. This is also what the
        # documentation recommends.  However, they are documented
        # to always be the same on Windows, where it causes
        # problems if we try to include both.
        script = """
        import sys
        if hasattr(sys, "base_exec_prefix"):
            sys.stdout.write(sys.base_exec_prefix)
        else:
            sys.stdout.write(sys.exec_prefix)
        """
    else
        script = """
        import sys
        if hasattr(sys, "base_exec_prefix"):
            sys.stdout.write(sys.base_prefix)
            sys.stdout.write(":")
            sys.stdout.write(sys.base_exec_prefix)
        else:
            sys.stdout.write(sys.prefix)
            sys.stdout.write(":")
            sys.stdout.write(sys.exec_prefix)
        """
        # https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHOME
    end
    return read(pythonenv(`$pyprogramname -c $script`), String)
end
# To support `venv` standard library (as well as `virtualenv`), we
# need to use `sys.base_prefix` and `sys.base_exec_prefix` here.
# Otherwise, initializing Python in `__init__` below fails with
# unrecoverable error:
#
#   Fatal Python error: initfsencoding: unable to load the file system codec
#   ModuleNotFoundError: No module named 'encodings'
#
# This is because `venv` does not symlink standard libraries like
# `virtualenv`.  For example, `lib/python3.X/encodings` does not
# exist.  Rather, `venv` relies on the behavior of Python runtime:
#
#   If a file named "pyvenv.cfg" exists one directory above
#   sys.executable, sys.prefix and sys.exec_prefix are set to that
#   directory and it is also checked for site-packages
#   --- https://docs.python.org/3/library/venv.html
#
# Thus, we need point `PYTHONHOME` to `sys.base_prefix` and
# `sys.base_exec_prefix`.  If the virtual environment is created by
# `virtualenv`, those `sys.base_*` paths point to the virtual
# environment.  Thus, above code supports both use cases.
#
# See also:
# * https://docs.python.org/3/library/venv.html
# * https://docs.python.org/3/library/site.html
# * https://docs.python.org/3/library/sys.html#sys.base_exec_prefix
# * https://github.com/JuliaPy/PyCall.jl/issues/410
