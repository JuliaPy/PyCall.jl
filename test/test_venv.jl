using PyCall, Compat, Compat.Test
using Compat: @info, @warn


@testset "fuzz PyCall._leak" begin
    N = 10000

    @testset "_leak(Cstring, ...)" begin
        for i in 1:N
            x = String(rand('A':'z', rand(1:1000)))
            y = Base.unsafe_string(PyCall._leak(Cstring, x))
            @test x == y
        end
    end

    @testset "_leak(Cwstring, ...)" begin
        for i in 1:N
            x = String(rand('A':'z', rand(1:1000)))
            a = Base.cconvert(Cwstring, x)
            ptr = PyCall._leak(a)
            z = unsafe_wrap(Array, ptr, size(a))
            @test z[end] == 0
            y = transcode(String, z)[1:end-1]
            @test x == y
        end
    end
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
    if VERSION < v"0.7.0-"
        setup_code = ""
    else
        setup_code = Base.load_path_setup_code()
    end
    code = """
    $setup_code
    using PyCall
    println(PyCall.pyimport("sys")[:executable])
    println(PyCall.pyimport("sys")[:exec_prefix])
    println(PyCall.pyimport("pip")[:__file__])
    """
    # Note that `pip` is just some arbitrary non-standard
    # library.  Using standard library like `os` does not work
    # because those files are not created.
    env = copy(ENV)
    env["PYCALL_JL_RUNTIME_PYTHON"] = newpython
    jlcmd = setenv(`$(Base.julia_cmd()) --startup-file=no -e $code`, env)
    if Compat.Sys.iswindows()
        # Marking the test broken in Windows.  It seems that
        # venv copies .dll on Windows and libpython check in
        # PyCall.__init__ detects that.
        @test_broken begin
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
    if Compat.Sys.which("virtualenv") === nothing
        @info "No virtualenv command. Skipping the test..."
    elseif Compat.Sys.which(pyname) === nothing
        @info "No $pyname command. Skipping the test..."
    else
        mktempdir() do path
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
    if Compat.Sys.isunix() && !startswith(PyCall.pyprogramname, "/usr/bin")
        @warn """
        Note that "venv activation" test does not work when PyCall is built
        with a Python executable created by `virtualenv` command.  You are
        using possibly non-system Python executable:
            $(PyCall.pyprogramname)

        Following commands may solve the failure (if any):
            julia> ENV["PYTHON"] = "/usr/bin/python3"
            pkg> build PyCall
            pkg> test PyCall
        """
        # Let's just warn it.  Not sure how to reliably detect it...
    end

    if PyCall.conda
        @info "Skip venv test with conda."
    elseif !success(PyCall.python_cmd(`-c "import venv"`))
        @info "Skip venv test since venv package is missing."
    else
        mktempdir() do path
            # Create a new virtual environment
            run(PyCall.python_cmd(`-m venv $path`))
            test_venv_has_python(path)
            test_venv_activation(path)
        end
    end
end
