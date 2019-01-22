using Test, PyCall

py"""
def mklist(*args, **kwargs):
    l = list(args)
    l.extend(kwargs.items())
    return l
"""
@testset "pycall!" begin
    pymklist = py"mklist"
    ret = PyNULL()

    function pycall_checks(res, pyf, RetType, args...; kwargs...)
        pycall_res = pycall(pyf, RetType, args...; kwargs...)
        res_converted = pycall!(res, pyf, RetType, args...; kwargs...)
        @test res_converted == pycall_res
        @test convert(RetType, res) == res_converted
        RetType != PyAny && @test res_converted isa RetType
    end

    @testset "basics" begin
        args = ("a", 2, 4.5)
        for RetType in (PyObject, PyAny, Tuple)
            pycall_checks(ret, pymklist, RetType, args...)
            GC.gc()
        end
    end

    @testset "kwargs" begin
        args = ("a", 2, 4.5)
        for RetType in (PyObject, PyAny, Tuple)
            pycall_checks(ret, pymklist, RetType, args...; b=19610, c="5")
            GC.gc()
        end
    end

end
