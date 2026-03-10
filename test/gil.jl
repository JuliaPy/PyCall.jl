using Test, PyCall
using Base.Threads

@testset "GIL / multi-threading" begin
    o = PyObject(randn(3,3))
    t = Threads.@spawn begin
        # Test that accessing `PyObject` across threads / tasks
        # does not immediately segfault (GIL is acquired correctly).
        iob = IOBuffer()
        println(iob, propertynames(o))
        str = String(take!(iob))
        return length(str)
    end
    @test fetch(t) != 0
end
