using Test, PyCall
using Base.Threads

@testset "GIL" begin

    @testset "basic multi-threading" begin
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

    @testset "finalizer deadlock" begin
        # Test that we avoid finalizer deadlocks like this:
        #   1. Task A holds the GIL
        #   2. Task B triggers a PyObject finalizer (e.g. via GC)
        #   3. Task A waits on Task B
        #   4. Task B cannot complete GC and Task A cannot advance -> deadlock

        o = PyObject(42)
        PyCall.pyincref(o)

        PyCall.@with_GIL begin
            t = Threads.@spawn begin
                finalize(o)
                return true
            end
            result = timedwait(() -> istaskdone(t), 5.0)
            @test result === :ok
            @test fetch(t) === true
            @test !isempty(PyCall._deferred_Py_DecRef)
        end
        @test isempty(PyCall._deferred_Py_DecRef)
    end

end
