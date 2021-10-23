module TestPyCallBuild

include(joinpath(dirname(@__FILE__), "..", "deps", "depsutils.jl"))
include(joinpath(dirname(@__FILE__), "..", "deps", "buildutils.jl"))

using Test

@testset "find_libpython" begin
    for python in ["python", "python2", "python3"]
        # TODO: In Windows, word size should also be checked.
        Sys.iswindows() && break
        if Sys.which(python) === nothing
            @info "$python not available; skipping test"
        else
            @test isfile(find_libpython(python)[2])
        end
    end

    # Test the case `find_libpython.py` does not print anything.  We
    # use the command `true` to mimic this case.
    if Sys.which("true") === nothing
        @info "no `true` command; skipping test"
    else
        let err, msg
            @test try
                find_libpython("true")
                false
            catch err
                err isa ErrorException
            end
            msg = sprint(showerror, err)
            @test occursin("Couldn't find libpython", msg)
            @test occursin("ENV[\"PYCALL_DEBUG_BUILD\"] = \"yes\"", msg)
        end
    end

    # Test the case `dlopen` failed to open the library.
    let err, msg
        @test try
            find_libpython("python"; _dlopen = (_...) -> error("dummy"))
            false
        catch err
            err isa ErrorException
        end
        msg = sprint(showerror, err)
        @test occursin("Couldn't find libpython", msg)
        @test occursin("ENV[\"PYCALL_DEBUG_BUILD\"] = \"yes\"", msg)
    end
end

end  # module
