# In this file, we figure out how to link to Python (surprisingly complicated)
# and generate a deps/deps.jl file with the libpython name and other information
# needed for static compilation of PyCall.

# As a result, if you switch to a different version or path of Python, you
# will probably need to re-run Pkg.build("PyCall").

using Compat, VersionParsing
import Conda, Compat.Libdl

struct UseCondaPython <: Exception end

#########################################################################

pyvar(python::AbstractString, mod::AbstractString, var::AbstractString) = chomp(read(pythonenv(`$python -c "import $mod; print($mod.$var)"`), String))

pyconfigvar(python::AbstractString, var::AbstractString) = pyvar(python, "distutils.sysconfig", "get_config_var('$var')")
pyconfigvar(python, var, default) = let v = pyconfigvar(python, var)
    v == "None" ? default : v
end

pysys(python::AbstractString, var::AbstractString) = pyvar(python, "sys", var)

#########################################################################

const dlprefix = Compat.Sys.iswindows() ? "" : "lib"

# print out extra info to help with remote debugging
const PYCALL_DEBUG_BUILD = "yes" == get(ENV, "PYCALL_DEBUG_BUILD", "no")

function exec_find_libpython(python::AbstractString, options)
    cmd = `$python $(joinpath(@__DIR__, "find_libpython.py")) $options`
    if PYCALL_DEBUG_BUILD
        cmd = `$cmd --verbose`
    end
    return readlines(pythonenv(cmd))
end

function show_dlopen_error(e)
    if PYCALL_DEBUG_BUILD
        println(stderr, "dlopen($libpath_lib) ==> ", e)
        # Using STDERR since find_libpython.py prints debugging
        # messages to STDERR too.
    end
end

# return libpython name, libpython pointer
function find_libpython(python::AbstractString)
    dlopen_flags = Libdl.RTLD_LAZY|Libdl.RTLD_DEEPBIND|Libdl.RTLD_GLOBAL

    libpaths = exec_find_libpython(python, `--list-all`)
    for lib in libpaths
        try
            return (Libdl.dlopen(lib, dlopen_flags), lib)
        catch e
            show_dlopen_error(e)
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
            libpython = Libdl.dlopen(lib, dlopen_flags)
            # Store the fullpath to libpython in deps.jl.  This makes
            # it easier for users to investigate Python setup
            # PyCall.jl trying to use.  It also helps PyJulia to
            # compare libpython.
            return (libpython, Libdl.dlpath(libpython))
        catch e
            show_dlopen_error(e)
        end
    end

    error("""
        Couldn't find libpython; check your PYTHON environment variable.

        The python executable we tried was $python (= version $v).
        Re-building with
            ENV["PYCALL_DEBUG_BUILD"] = "yes"
        may provide extra information for why it failed.
        """)
end

#########################################################################

include("depsutils.jl")

#########################################################################

# A couple of key strings need to be stored as constants so that
# they persist throughout the life of the program.  In Python 3,
# they need to be wchar_t* data.
wstringconst(s) = string("Base.cconvert(Cwstring, \"", escape_string(s), "\")")

# we write configuration files only if they change, both
# to prevent unnecessary recompilation and to minimize
# problems in the unlikely event of read-only directories.
function writeifchanged(filename, str)
    if !isfile(filename) || read(filename, String) != str
        Compat.@info string(abspath(filename), " has been updated")
        write(filename, str)
    else
        Compat.@info string(abspath(filename), " has not changed")
    end
end

# return the first arg that exists in the PATH
function whichfirst(args...)
    for x in args
        if Compat.Sys.which(x) !== nothing
            return x
        end
    end
    return ""
end

const prefsfile = VERSION < v"0.7" ? "PYTHON" : joinpath(first(DEPOT_PATH), "prefs", "PyCall")
mkpath(dirname(prefsfile))

try # make sure deps.jl file is removed on error
    python = try
        let py = get(ENV, "PYTHON", isfile(prefsfile) ? readchomp(prefsfile) :
                     (Compat.Sys.isunix() && !Compat.Sys.isapple()) || Sys.ARCH ∉ (:i686, :x86_64) ?
                     whichfirst("python3", "python") : "Conda"),
            vers = isempty(py) || py == "Conda" ? v"0.0" : vparse(pyconfigvar(py,"VERSION","0.0"))
            if vers < v"2.7"
                if isempty(py) || py == "Conda"
                    throw(UseCondaPython())
                else
                    error("Python version $vers < 2.7 is not supported")
                end
            end

            # check word size of Python via sys.maxsize, since a common error
            # on Windows is to link a 64-bit Julia to a 32-bit Python.
            pywordsize = parse(UInt64, pysys(py, "maxsize")) > (UInt64(1)<<32) ? 64 : 32
            if pywordsize != Sys.WORD_SIZE
                error("$py is $(pywordsize)-bit, but Julia is $(Sys.WORD_SIZE)-bit")
            end

            py
        end
    catch e1
        if Sys.ARCH in (:i686, :x86_64)
            if isa(e1, UseCondaPython)
                Compat.@info string("Using the Python distribution in the Conda package by default.\n",
                     "To use a different Python version, set ENV[\"PYTHON\"]=\"pythoncommand\" and re-run Pkg.build(\"PyCall\").")
            else
                Compat.@info string( "No system-wide Python was found; got the following error:\n",
                      "$e1\nusing the Python distribution in the Conda package")
            end
            abspath(Conda.PYTHONDIR, "python" * (Compat.Sys.iswindows() ? ".exe" : ""))
        else
            error("No system-wide Python was found; got the following error:\n",
                  "$e1")
        end
    end

    use_conda = dirname(python) == abspath(Conda.PYTHONDIR)
    if use_conda
        Conda.add("numpy")
    end

    (libpython, libpy_name) = find_libpython(python)
    programname = pysys(python, "executable")

    # Get PYTHONHOME, either from the environment or from Python
    # itself (if it is not in the environment or if we are using Conda)
    PYTHONHOME = if !haskey(ENV, "PYTHONHOME") || use_conda
        pythonhome_of(python)
    else
        ENV["PYTHONHOME"]
    end

    # cache the Python version as a Julia VersionNumber
    pyversion = vparse(pyvar(python, "platform", "python_version()"))

    Compat.@info "PyCall is using $python (Python $pyversion) at $programname, libpython = $libpy_name"

    if pyversion < v"2.7"
        error("Python 2.7 or later is required for PyCall")
    end

    writeifchanged("deps.jl", """
    const python = "$(escape_string(python))"
    const libpython = "$(escape_string(libpy_name))"
    const pyprogramname = "$(escape_string(programname))"
    const wpyprogramname = $(wstringconst(programname))
    const pyversion_build = $(repr(pyversion))
    const PYTHONHOME = "$(escape_string(PYTHONHOME))"
    const wPYTHONHOME = $(wstringconst(PYTHONHOME))

    "True if we are using the Python distribution in the Conda package."
    const conda = $use_conda
    """)

    # Make subsequent builds (e.g. Pkg.update) use the same Python by default:
    writeifchanged(prefsfile, use_conda ? "Conda" : isfile(programname) ? programname : python)

    #########################################################################

catch

    # remove deps.jl (if it exists) on an error, so that PyCall will
    # not load until it is properly configured.
    isfile("deps.jl") && rm("deps.jl")
    rethrow()

end
