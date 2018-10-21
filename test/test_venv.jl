using PyCall, Compat, Compat.Test
using Compat: @info


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


@testset "venv activation" begin
    if PyCall.conda
        @info "Skip venv test with conda."
    elseif !success(PyCall.python_cmd(`-c "import venv"`))
        @info "Skip venv test since venv package is missing."
    else
        mktempdir() do path
            # Create a new virtualenv
            run(PyCall.python_cmd(`-m venv $path`))
            newpython = joinpath(path, "bin", "python")
            if Compat.Sys.iswindows()
                newpython *= ".exe"
            end
            @test isfile(newpython)

            # Run a fresh Julia process with new Python environment
            if VERSION < v"0.7.0-"
                setup_code = ""
            else
                setup_code = Base.load_path_setup_code()
            end
            code = """
            $setup_code
            using PyCall
            print(PyCall.pyimport("sys")[:executable])
            """
            env = copy(ENV)
            env["PYCALL_JL_RUNTIME_PYTHON"] = newpython
            jlcmd = setenv(`$(Base.julia_cmd()) -e $code`, env)
            output = read(jlcmd, String)
            @test newpython == output
        end
    end
end
