# Included from build.jl, ../test/test_build.jl and ../src/PyCall.jl

import Libdl

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
const __buf_programname = UInt8[]
const __buf_pythonhome = UInt8[]

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
    copyto!(resize!(dest, length(s) + 1), s)
    dest[length(s) + 1] = 0
    return pointer(dest)
end

function _preserveas!(dest::Vector{UInt8}, ::Type{Cwstring}, x::AbstractString)
    s = reinterpret(UInt8, Base.cconvert(Cwstring, x))
    copyto!(resize!(dest, length(s)), s)
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

