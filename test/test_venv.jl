using PyCall, Test

"""Gets temp directory, possibly with user-set environment variable"""
function _mktempdir(f::Function, parent=nothing; kwargs...)
    if parent === nothing && haskey(ENV, "_TMP_DIR")
        return mktempdir(f, ENV["_TMP_DIR"]; kwargs...)
    end
    if parent === nothing
        parent = tempdir()
    end
    return mktempdir(f, parent; kwargs...)
end


function test_venv_has_python(path)
    newpython = PyCall.python_cmd(venv=path).exec[1]
    if !isfile(newpython)
        @info """
        Python executable $newpython does not exists.
        This directory contains only the following files:
        $(join(readdir(dirname(newpython)), '\n'))
        """
    end
    @test isfile(newpython)
end


function test_venv_activation(path)
    newpython = PyCall.python_cmd(venv=path).exec[1]

    # Run a fresh Julia process with new Python environment
    code = """
    $(Base.load_path_setup_code())
    using PyCall
    println(PyCall.pyimport("sys").executable)
    println(PyCall.pyimport("sys").exec_prefix)
    println(PyCall.pyimport("pip").__file__)
    """
    # Note that `pip` is just some arbitrary non-standard
    # library.  Using standard library like `os` does not work
    # because those files are not created.
    env = copy(ENV)
    env["PYCALL_JL_RUNTIME_PYTHON"] = newpython
    jlcmd = setenv(`$(Base.julia_cmd()) --startup-file=no -e $code`, env)
    if Sys.iswindows()
        # Marking the test broken in Windows.  It seems that
        # venv copies .dll on Windows and libpython check in
        # PyCall.__init__ detects that.
        @test begin
            output = read(jlcmd, String)
            sys_executable, exec_prefix, mod_file = split(output, "\n")
            newpython == sys_executable
        end
    else
        output = read(jlcmd, String)
        sys_executable, exec_prefix, mod_file = split(output, "\n")
        @test newpython == sys_executable
        @test startswith(exec_prefix, path)
        @test startswith(mod_file, path)
    end
end


@testset "virtualenv activation" begin
    pyname = "python$(pyversion.major).$(pyversion.minor)"
    if Sys.which("virtualenv") === nothing
        @info "No virtualenv command. Skipping the test..."
    elseif Sys.which(pyname) === nothing
        @info "No $pyname command. Skipping the test..."
    else
        _mktempdir() do tmppath
            if PyCall.pyversion.major == 2
                path = joinpath(tmppath, "kind")
            else
                path = joinpath(tmppath, "ϵνιℓ")
            end
            run(`virtualenv --python=$pyname $path`)
            test_venv_has_python(path)

            newpython = PyCall.python_cmd(venv=path).exec[1]
            venv_libpython = PyCall.find_libpython(newpython)
            if venv_libpython != PyCall.libpython
                @info """
                virtualenv created an environment with incompatible libpython:
                    $venv_libpython
                """
                return
            end

            test_venv_activation(path)
        end
    end
end


@testset "venv activation" begin
    # In case PyCall is built with a Python executable created by
    # `virtualenv`, let's try to find the original Python executable.
    # Otherwise, `venv` does not work with this Python executable:
    # https://bugs.python.org/issue30811
    sys = PyCall.pyimport("sys")
    if hasproperty(sys, :real_prefix)
        # sys.real_prefix is set by virtualenv and does not exist in
        # standard Python:
        # https://github.com/pypa/virtualenv/blob/16.0.0/virtualenv_embedded/site.py#L554
        candidates = [
            PyCall.venv_python(sys.real_prefix, "$(pyversion.major).$(pyversion.minor)"),
            PyCall.venv_python(sys.real_prefix, "$(pyversion.major)"),
            PyCall.venv_python(sys.real_prefix),
            PyCall.pyprogramname,  # must exists
        ]
        python = candidates[findfirst(isfile, candidates)]
    else
        python = PyCall.pyprogramname
    end

    if PyCall.conda
        @info "Skip venv test with conda."
    elseif !success(PyCall.python_cmd(`-c "import venv"`, python=python))
        @info "Skip venv test since venv package is missing."
    else
        _mktempdir() do tmppath
            if PyCall.pyversion.major == 2
                path = joinpath(tmppath, "kind")
            else
                path = joinpath(tmppath, "ϵνιℓ")
            end
            # Create a new virtual environment
            run(PyCall.python_cmd(`-m venv $path`, python=python))
            test_venv_has_python(path)
            test_venv_activation(path)
        end
    end
end
