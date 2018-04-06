using Compat.Test, PyCall

@testset "pair call syntax" begin
    pyarr = pyimport("array")["array"]
    arrpyo = pyarr=>PyObject
    arrpytpl = pyarr=>Tuple{Vararg{Float64}}
    arrpyvec = pyarr=>Vector{Float64}
    arrpyany = pyarr=>PyAny
    @test typeof(arrpyo("d", 1.0:10.0)) == PyObject
    @test arrpytpl("d", 1.0:10.0) isa Tuple{Vararg{Float64}}
    @test typeof(arrpytpl("d", 1.0:10.0)) == NTuple{10, Float64}
    @test typeof(arrpyvec("d", 1.0:10.0)) == Vector{Float64}
    # broken with __array_interface__ but not with buffer protocol
    @test_broken typeof(arrpyany("d", 1.0:10.0)) == Vector{Float64}
end
