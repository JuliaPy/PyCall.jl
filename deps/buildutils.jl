# Included from build.jl and ../test/test_build.jl

using VersionParsing
import Conda, Libdl

pyvar(python::AbstractString, mod::AbstractString, var::AbstractString) = chomp(read(pythonenv(`$python -c "import $mod; print($mod.$(var))"`), String))

function pyconfigvar(python::AbstractString, var::AbstractString)
    try
       pyvar(python, "sysconfig", "get_config_var('$(var)')")
    catch e
        emsg = sprint(showerror, e)
        @warn "Encountered error on using `sysconfig`: $emsg. Falling back to `distutils.sysconfig`."
        pyvar(python, "distutils.sysconfig", "get_config_var('$(var)')")
    end
end
pyconfigvar(python, var, default) = let v = pyconfigvar(python, var)
    v == "None" ? default : v
end

pysys(python::AbstractString, var::AbstractString) = pyvar(python, "sys", var)

#########################################################################

# print out extra info to help with remote debugging
const PYCALL_DEBUG_BUILD = "yes" == get(ENV, "PYCALL_DEBUG_BUILD", "no")

function exec_find_libpython(python::AbstractString, options)
    # Do not inline `@__DIR__` into the backticks to expand correctly.
    # See: https://github.com/JuliaLang/julia/issues/26323
    script = joinpath(@__DIR__, "find_libpython.py")
    cmd = `$python $script $options`
    if PYCALL_DEBUG_BUILD
        cmd = `$cmd --verbose`
    end
    return readlines(pythonenv(cmd))
end

function show_dlopen_error(lib, e)
    if PYCALL_DEBUG_BUILD
        println(stderr, "dlopen($lib) ==> ", e)
        # Using STDERR since find_libpython.py prints debugging
        # messages to STDERR too.
    end
end

# return libpython name, libpython pointer
function find_libpython(python::AbstractString; _dlopen = Libdl.dlopen)
    dlopen_flags = Libdl.RTLD_LAZY|Libdl.RTLD_DEEPBIND|Libdl.RTLD_GLOBAL

    libpaths = exec_find_libpython(python, `--list-all`)
    for lib in libpaths
        try
            return (_dlopen(lib, dlopen_flags), lib)
        catch e
            show_dlopen_error(lib, e)
        end
    end

    # Try all candidate libpython names and let Libdl find the path.
    # We do this *last* because the libpython in the system
    # library path might be the wrong one if multiple python
    # versions are installed (we prefer the one in LIBDIR):
    libs = exec_find_libpython(python, `--candidate-names`)
    for lib in libs
        lib = splitext(lib)[1]
        try
            libpython = _dlopen(lib, dlopen_flags)
            # Store the fullpath to libpython in deps.jl.  This makes
            # it easier for users to investigate Python setup
            # PyCall.jl trying to use.  It also helps PyJulia to
            # compare libpython.
            return (libpython, Libdl.dlpath(libpython))
        catch e
            show_dlopen_error(lib, e)
        end
    end

    v = pyconfigvar(python, "VERSION", "unknown")
    error("""
        Couldn't find libpython; check your PYTHON environment variable.

        The python executable we tried was $python (= version $v).
        Re-building with
            ENV["PYCALL_DEBUG_BUILD"] = "yes"
        may provide extra information for why it failed.
        """)
end
