using Test
const desired_version_str = ARGS[1]  # e.g., "2.7", "3.10.0", "3.x"
include("../deps/deps.jl")
@testset "pyversion_build ≈ $desired_version_str" begin
    v = desired_version_str
    while endswith(v, ".x")
        v = v[1:end-2]
    end
    desired_version = VersionNumber(v)
    @test desired_version.major == pyversion_build.major
    if desired_version.minor != 0
        @test desired_version.minor == pyversion_build.minor
    end
    if desired_version.patch != 0
        @test desired_version.patch == pyversion_build.patch
    end
end
