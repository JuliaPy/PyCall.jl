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

# Need to set PythonHome before calling GetVersion to avoid warning (#299).
# Unfortunately, this poses something of a chicken-and-egg problem because
# we need to know the Python version to set PythonHome via the API.  Note
# that the string (or array) passed to Py_SetPythonHome needs to be a
# constant that lasts for the lifetime of the program, which is why we
# can't use Cwstring here (since that creates a temporary copy).
function Py_SetPythonHome(libpy, PYTHONHOME, wPYTHONHOME, pyversion)
    if !isempty(PYTHONHOME)
        if pyversion.major < 3
            ccall(Libdl.dlsym(libpy, :Py_SetPythonHome), Cvoid, (Cstring,), PYTHONHOME)
        else
            ccall(Libdl.dlsym(libpy, :Py_SetPythonHome), Cvoid, (Ptr{Cwchar_t},), wPYTHONHOME)
        end
    end
end

# need to be able to get the version before Python is initialized
Py_GetVersion(libpy) = unsafe_string(ccall(Libdl.dlsym(libpy, :Py_GetVersion), Ptr{UInt8}, ()))
