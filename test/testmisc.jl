using Compat.Test, PyCall

PyCall.@npyinitialize # see #481

@testset "pair call syntax" begin
    np = pyimport("numpy")
    arrpyo = np["array"]=>PyObject
    arrpytpl = np["array"]=>Tuple{Vararg{Int}}
    arrpyvec = np["array"]=>Vector{Int}
    arrpyany = np["array"]=>PyAny
    @test typeof(arrpyo(1:10)) == PyObject
    @test arrpytpl(1:10) isa Tuple{Vararg{Int}}
    @test typeof(arrpytpl(1:10)) == NTuple{10, Int}
    @test typeof(arrpyvec(1:10)) == Vector{Int}
    @test typeof(arrpyany(1:10)) == Vector{Int}
end
