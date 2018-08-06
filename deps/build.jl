# In this file, we figure out how to link to Python (surprisingly complicated)
# and generate a deps/deps.jl file with the libpython name and other information
# needed for static compilation of PyCall.

# As a result, if you switch to a different version or path of Python, you
# will probably need to re-run Pkg.build("PyCall").

using Compat, VersionParsing
import Conda, Compat.Libdl

struct UseCondaPython <: Exception end

#########################################################################

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

pyvar(python::AbstractString, mod::AbstractString, var::AbstractString) = chomp(read(pythonenv(`$python -c "import $mod; print($mod.$var)"`), String))

pyconfigvar(python::AbstractString, var::AbstractString) = pyvar(python, "distutils.sysconfig", "get_config_var('$var')")
pyconfigvar(python, var, default) = let v = pyconfigvar(python, var)
    v == "None" ? default : v
end

pysys(python::AbstractString, var::AbstractString) = pyvar(python, "sys", var)

#########################################################################

const dlprefix = Compat.Sys.iswindows() ? "" : "lib"

# return libpython name, libpython pointer
function find_libpython(python::AbstractString)
    # it is ridiculous that it is this hard to find the name of libpython
    v = pyconfigvar(python,"VERSION","")
    libs = [ dlprefix*"python"*v, dlprefix*"python" ]
    lib = pyconfigvar(python, "LIBRARY")
    lib != "None" && pushfirst!(libs, splitext(lib)[1])
    lib = pyconfigvar(python, "LDLIBRARY")
    lib != "None" && pushfirst!(pushfirst!(libs, basename(lib)), lib)
    libs = unique(libs)

    # it is ridiculous that it is this hard to find the path of libpython
    libpaths = [pyconfigvar(python, "LIBDIR"),
                (Compat.Sys.iswindows() ? dirname(pysys(python, "executable")) : joinpath(dirname(dirname(pysys(python, "executable"))), "lib"))]
    if Compat.Sys.isapple()
        push!(libpaths, pyconfigvar(python, "PYTHONFRAMEWORKPREFIX"))
    end

    # `prefix` and `exec_prefix` are the path prefixes where python should look for python only and compiled libraries, respectively.
    # These are also changed when run in a virtualenv.
    exec_prefix = pysys(python, "exec_prefix")

    push!(libpaths, exec_prefix)
    push!(libpaths, joinpath(exec_prefix, "lib"))

    error_strings = String[]

    # TODO: other paths? python-config output? pyconfigvar("LDFLAGS")?

    # find libpython (we hope):
    for lib in libs
        for libpath in libpaths
            libpath_lib = joinpath(libpath, lib)
            if isfile(libpath_lib*"."*Libdl.dlext)
                try
                    return (Libdl.dlopen(libpath_lib,
                                         Libdl.RTLD_LAZY|Libdl.RTLD_DEEPBIND|Libdl.RTLD_GLOBAL),
                            libpath_lib)
                catch e
                    push!(error_strings, string("dlopen($libpath_lib) ==> ", e))
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
        catch e
            push!(error_strings, string("dlopen($lib) ==> ", e))
        end
    end

    if "yes" == get(ENV, "PYCALL_DEBUG_BUILD", "no") # print out extra info to help with remote debugging
        println(stderr, "------------------------------------- exceptions -----------------------------------------")
        for s in error_strings
            print(s, "\n\n")
        end
        println(stderr, "---------------------------------- get_config_vars ---------------------------------------")
        print(stderr, read(`python -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_vars())"`, String))
        println(stderr, "--------------------------------- directory contents -------------------------------------")
        for libpath in libpaths
            if isdir(libpath)
                print(libpath, ":\n")
                for file in readdir(libpath)
                    if occursin("pyth", file)
                        println("    ", file)
                    end
                end
            end
        end
        println(stderr, "------------------------------------------------------------------------------------------")
    end

    error("""
        Couldn't find libpython; check your PYTHON environment variable.

        The python executable we tried was $python (= version $v);
        the library names we tried were $libs
        and the library paths we tried were $libpaths""")
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

try # make sure deps.jl file is removed on error
    python = try
        let py = get(ENV, "PYTHON", isfile("PYTHON") ? readchomp("PYTHON") :
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
        # PYTHONHOME tells python where to look for both pure python
        # and binary modules.  When it is set, it replaces both
        # `prefix` and `exec_prefix` and we thus need to set it to
        # both in case they differ. This is also what the
        # documentation recommends.  However, they are documented
        # to always be the same on Windows, where it causes
        # problems if we try to include both.
        exec_prefix = pysys(python, "exec_prefix")
        Compat.Sys.iswindows() ? exec_prefix : pysys(python, "prefix") * ":" * exec_prefix
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
    writeifchanged("PYTHON", use_conda ? "Conda" : isfile(programname) ? programname : python)

    #########################################################################

catch

    # remove deps.jl (if it exists) on an error, so that PyCall will
    # not load until it is properly configured.
    isfile("deps.jl") && rm("deps.jl")
    rethrow()

end
