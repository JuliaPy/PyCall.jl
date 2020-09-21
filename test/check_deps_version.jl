using Test
const desired_version = VersionNumber(ARGS[1])
include("../deps/deps.jl")
@testset "pyversion_build ≈ $desired_version" begin
    @test desired_version.major == pyversion_build.major
    @test desired_version.minor == pyversion_build.minor
    if desired_version.patch != 0
        @test desired_version.patch == pyversion_build.patch
    end
end
