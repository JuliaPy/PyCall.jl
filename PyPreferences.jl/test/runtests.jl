using PyPreferences
using Test

@testset "PyPreferences.jl" begin
    # Write your tests here.
end

if lowercase(get(ENV, "JULIA_PKGEVAL", "false")) != "true"
    include("test_venv.jl")
end
