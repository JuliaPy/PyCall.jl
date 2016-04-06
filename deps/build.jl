# In this file, we figure out how to link to Python (surprisingly complicated)
# and generate a deps/deps.jl file with the libpython name and other information
# needed for static compilation of PyCall.

# As a result, if you switch to a different version or path of Python, you
# will probably need to re-run Pkg.build("PyCall").

# remove deps.jl if it exists, in case build.jl fails
isfile("deps.jl") && rm("deps.jl")

using Compat
import Conda

PYTHONIOENCODING = get(ENV, "PYTHONIOENCODING", nothing)
PYTHONHOME = get(ENV, "PYTHONHOME", nothing)

try # save/restore environment vars

# set PYTHONIOENCODING when running python executable, so that
# we get UTF-8 encoded text as output (this is not the default on Windows).
ENV["PYTHONIOENCODING"] = "UTF-8"

#########################################################################

pyconfigvar(python::AbstractString, var::AbstractString) = chomp(readstring(`$python -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('$var'))"`))
pyconfigvar(python, var, default) = let v = pyconfigvar(python, var)
    v == "None" ? default : v
end

pysys(python::AbstractString, var::AbstractString) = chomp(readstring(`$python -c "import sys; print(sys.$var)"`))

#########################################################################

const dlprefix = @windows? "" : "lib"

# return libpython name, libpython pointer
function find_libpython(python::AbstractString)
    # it is ridiculous that it is this hard to find the name of libpython
    v = pyconfigvar(python,"VERSION","")
    libs = [ dlprefix*"python"*v*"."*Libdl.dlext, dlprefix*"python."*Libdl.dlext ]
    lib = pyconfigvar(python, "LIBRARY")
    lib != "None" && unshift!(libs, splitext(lib)[1]*"."*Libdl.dlext)
    lib = pyconfigvar(python, "LDLIBRARY")
    lib != "None" && unshift!(unshift!(libs, basename(lib)), lib)
    libs = unique(libs)

    # it is ridiculous that it is this hard to find the path of libpython
    libpaths = [pyconfigvar(python, "LIBDIR"),
                (@windows ? dirname(pysys(python, "executable")) : joinpath(dirname(dirname(pysys(python, "executable"))), "lib"))]
    @osx_only push!(libpaths, pyconfigvar(python, "PYTHONFRAMEWORKPREFIX"))

    # `prefix` and `exec_prefix` are the path prefixes where python should look for python only and compiled libraries, respectively.
    # These are also changed when run in a virtualenv.
    exec_prefix = pysys(python, "exec_prefix")

    push!(libpaths, exec_prefix)
    push!(libpaths, joinpath(exec_prefix, "lib"))

    if !haskey(ENV, "PYTHONHOME")
        # PYTHONHOME tells python where to look for both pure python
        # and binary modules.  When it is set, it replaces both
        # `prefix` and `exec_prefix` and we thus need to set it to
        # both in case they differ. This is also what the
        # documentation recommends.  However, they are documented
        # to always be the same on Windows, where it causes
        # problems if we try to include both.
        ENV["PYTHONHOME"] = @windows? exec_prefix : pysys(python, "prefix") * ":" * exec_prefix
        # Unfortunately, setting PYTHONHOME screws up Canopy's Python distro?
        try
            run(pipeline(`$python -c "import site"`, stdout=DevNull, stderr=DevNull))
        catch
            pop!(ENV, "PYTHONHOME")
        end
    end

    # TODO: other paths? python-config output? pyconfigvar("LDFLAGS")?

    # find libpython (we hope):
    for lib in libs
        for libpath in libpaths
            libpath_lib = joinpath(libpath, lib)
            if isfile(libpath_lib)
                try
                    return (Libdl.dlopen(libpath_lib,
                                         Libdl.RTLD_LAZY|Libdl.RTLD_DEEPBIND|Libdl.RTLD_GLOBAL),
                            libpath_lib)
                end
            end
        end
    end

    # We do this *last* because the libpython in the system
    # library path might be the wrong one if multiple python
    # versions are installed (we prefer the one in LIBDIR):
    for lib in libs
        lib = splitext(lib)[1]
        try
            return (Libdl.dlopen(lib, Libdl.RTLD_LAZY|Libdl.RTLD_DEEPBIND|Libdl.RTLD_GLOBAL),
                    lib)
        end
    end
    error("Couldn't find libpython; check your PYTHON environment variable")
end

#########################################################################

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

#########################################################################

# need to be able to get the version before Python is initialized
Py_GetVersion(libpy) = bytestring(ccall(Libdl.dlsym(libpy, :Py_GetVersion), Ptr{UInt8}, ()))

const python = try
    let py = get(ENV, "PYTHON", isfile("PYTHON") ? readchomp("PYTHON") : "python"), vers = convert(VersionNumber, pyconfigvar(py,"VERSION","0.0"))
        if vers < v"2.7"
            error("Python version $vers < 2.7 is not supported")
        end
        py
    end
catch e1
    info( "No system-wide Python was found; got the following error:\n",
          "$e1\nusing the Python distribution in the Conda package")
    abspath(Conda.PYTHONDIR, "python" * (@windows? ".exe" : ""))
end

use_conda = dirname(python) == abspath(Conda.PYTHONDIR)
if use_conda
    Conda.add("numpy")
end

const (libpython, libpy_name) = find_libpython(python)
const programname = pysys(python, "executable")

# cache the Python version as a Julia VersionNumber
const pyversion = convert(VersionNumber, split(Py_GetVersion(libpython))[1])

info("PyCall is using $python (Python $pyversion) at $programname, libpython = $libpy_name")

if pyversion < v"2.7"
    error("Python 2.7 or later is required for PyCall")
end

# PyUnicode_* may actually be a #define for another symbol, so
# we cache the correct dlsym
const PyUnicode_AsUTF8String =
    findsym(libpython, :PyUnicode_AsUTF8String, :PyUnicodeUCS4_AsUTF8String, :PyUnicodeUCS2_AsUTF8String)
const PyUnicode_DecodeUTF8 =
    findsym(libpython, :PyUnicode_DecodeUTF8, :PyUnicodeUCS4_DecodeUTF8, :PyUnicodeUCS2_DecodeUTF8)

# Python 2/3 compatibility: cache symbols for renamed functions
if hassym(libpython, :PyString_FromStringAndSize)
    const PyString_FromStringAndSize = :PyString_FromStringAndSize
    const PyString_AsStringAndSize = :PyString_AsStringAndSize
    const PyString_Size = :PyString_Size
    const PyString_Type = :PyString_Type
else
    const PyString_FromStringAndSize = :PyBytes_FromStringAndSize
    const PyString_AsStringAndSize = :PyBytes_AsStringAndSize
    const PyString_Size = :PyBytes_Size
    const PyString_Type = :PyBytes_Type
end
if hassym(libpython, :PyInt_Type)
    const PyInt_Type = :PyInt_Type
    const PyInt_FromSize_t = :PyInt_FromSize_t
    const PyInt_FromSsize_t = :PyInt_FromSsize_t
    const PyInt_AsSsize_t = :PyInt_AsSsize_t
else
    const PyInt_Type = :PyLong_Type
    const PyInt_FromSize_t = :PyLong_FromSize_t
    const PyInt_FromSsize_t = :PyLong_FromSsize_t
    const PyInt_AsSsize_t = :PyLong_AsSsize_t
end

# hashes changed from long to intptr_t in Python 3.2
const Py_hash_t = pyversion < v"3.2" ? Clong:Int

# whether to use unicode for strings by default, ala Python 3
const pyunicode_literals = pyversion >= v"3.0"

# some arguments changed from char* to wchar_t* in Python 3
pystring = pyversion.major < 3 ? "bytestring" : "wstring"

open("deps.jl", "w") do f
    print(f, """
          const python = "$(escape_string(python))"
          const libpython = "$(escape_string(libpy_name))"
          const pyprogramname = $pystring("$(escape_string(programname))")
          const pyversion_build = $(repr(pyversion))
          const PYTHONHOME = $pystring("$(escape_string(get(ENV, "PYTHONHOME", "")))")

          "True if we are using the Python distribution in the Conda package."
          const conda = $use_conda

          const PyUnicode_AsUTF8String = :$PyUnicode_AsUTF8String
          const PyUnicode_DecodeUTF8 = :$PyUnicode_DecodeUTF8

          const PyString_FromStringAndSize = :$PyString_FromStringAndSize
          const PyString_AsStringAndSize = :$PyString_AsStringAndSize
          const PyString_Size = :$PyString_Size
          const PyString_Type = :$PyString_Type
          const PyInt_Type = :$PyInt_Type
          const PyInt_FromSize_t = :$PyInt_FromSize_t
          const PyInt_FromSsize_t = :$PyInt_FromSsize_t
          const PyInt_AsSsize_t = :$PyInt_AsSsize_t

          const Py_hash_t = $Py_hash_t

          const pyunicode_literals = $pyunicode_literals
          """)
end

# Make subsequent builds (e.g. Pkg.update) use the same Python by default:
open("PYTHON", "w") do f
    println(f, isfile(programname) ? programname : python)
end

#########################################################################

finally # restore env vars

PYTHONIOENCODING != nothing ? (ENV["PYTHONIOENCODING"] = PYTHONIOENCODING) : pop!(ENV, "PYTHONIOENCODING")
PYTHONHOME != nothing ? (ENV["PYTHONHOME"] = PYTHONHOME) : pop!(ENV, "PYTHONHOME", "")

end
