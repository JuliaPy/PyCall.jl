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

    @testset "GIL + GC safepoint deadlock" begin
        done = Threads.Atomic{Bool}(false)

        # If not protected by a (GC-aware) Julia-level lock, it is possible
        # to deadlock with the GC + GIL:
        #   1. Task A holds the GIL
        #   2. Task B waits to access the GIL
        #   3. Task A triggers GC
        #   4. Task B never reaches a safepoint, due to waiting for the GIL.
        task1 = Threads.@spawn begin
            while !done[]
                PyCall.@with_GIL begin
                    for _ in 1:10_000
                        PyObject(0)
                    end
                end
            end
        end

        task2 = Threads.@spawn begin
            while !done[]
                PyCall.@with_GIL begin
                    for _ in 1:10_000
                        PyObject(0)
                    end
                end
                GC.gc(true)
            end
        end

        result = timedwait(() -> istaskdone(task1) || istaskdone(task2), 3.0)
        done[] = true
        timedwait(() -> istaskdone(task1) && istaskdone(task2), 5.0)
        @test result === :timed_out
    end

end
