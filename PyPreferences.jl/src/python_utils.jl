using Libdl: Libdl
using VersionParsing: vparse
using Conda: Conda

conda_python_fullpath() =
    abspath(Conda.PYTHONDIR, "python" * (Sys.iswindows() ? ".exe" : ""))

# Fix the environment for running `python`, and setts IO encoding to UTF-8.
# If cmd is the Conda python, then additionally removes all PYTHON* and
# CONDA* environment variables.
function pythonenv(cmd::Cmd)
    @assert cmd.env === nothing  # TODO: handle non-nothing case
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

pyvar(python::AbstractString, mod::AbstractString, var::AbstractString) =
    chomp(read(pythonenv(`$python -c "import $mod; print($mod.$(var))"`), String))

pyconfigvar(python::AbstractString, var::AbstractString) =
    pyvar(python, "distutils.sysconfig", "get_config_var('$(var)')")
pyconfigvar(python, var, default) =
    let v = pyconfigvar(python, var)
        v == "None" ? default : v
    end

function pythonhome_of(pyprogramname::AbstractString)
    if Sys.iswindows()
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

python_version_of(python) = vparse(pyvar(python, "platform", "python_version()"))

function find_libpython_py_path()

    return joinpath(@__DIR__, "find_libpython.py")
end

function exec_find_libpython(python::AbstractString, options, verbose::Bool)
    # Do not inline `@__DIR__` into the backticks to expand correctly.
    # See: https://github.com/JuliaLang/julia/issues/26323
    script = find_libpython_py_path()
    cmd = `$python $script $options`
    if verbose
        cmd = `$cmd --verbose`
    end
    return readlines(pythonenv(cmd))
end

# return libpython path, libpython pointer
function find_libpython(
    python::AbstractString;
    _dlopen = Libdl.dlopen,
    verbose::Bool = false,
)
    dlopen_flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL

    libpaths = exec_find_libpython(python, `--list-all`, verbose)
    for lib in libpaths
        try
            return (lib, _dlopen(lib, dlopen_flags))
        catch e
            @warn "Failed to `dlopen` $lib" exception = (e, catch_backtrace())
        end
    end
    @warn """
    Python (`find_libpython.py`) failed to find `libpython`.
    Falling back to `Libdl`-based discovery.
    """

    # Try all candidate libpython names and let Libdl find the path.
    # We do this *last* because the libpython in the system
    # library path might be the wrong one if multiple python
    # versions are installed (we prefer the one in LIBDIR):
    libs = exec_find_libpython(python, `--candidate-names`, verbose)
    for lib in libs
        lib = splitext(lib)[1]
        try
            libpython = _dlopen(lib, dlopen_flags)
            return (Libdl.dlpath(libpython), libpython)
        catch e
            @debug "Failed to `dlopen` $lib" exception = (e, catch_backtrace())
        end
    end

    return nothing, nothing
end
