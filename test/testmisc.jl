using Compat.Test, PyCall

@testset "pair call syntax" begin
    pylist = pybuiltin("list")
    listpyo = pylist=>PyObject
    listpytpl = pylist=>Tuple{Vararg{Float64}}
    listpyvec = pylist=>Vector{Float64}
    listpyany = pylist=>PyAny
    @test typeof(listpyo(1.0:10.0)) == PyObject
    @test listpytpl(1.0:10.0) isa Tuple{Vararg{Float64}}
    @test typeof(listpytpl(1.0:10.0)) == NTuple{10, Float64}
    @test typeof(listpyvec(1.0:10.0)) == Vector{Float64}
    @test typeof(listpyany(1.0:10.0)) == Vector{Float64}
end
