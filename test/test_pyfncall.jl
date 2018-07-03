using Compat.Test, PyCall

py"""
def mklist(*args):
    return list(args)
"""
@testset "pycall!" begin
    pymklist = py"mklist"
    res = PyNULL()

    function pycall_checks(res, pyf, RetType, args...)
        pycall_res = pycall(pyf, RetType, args...)
        res_converted = pycall!(res, pyf, RetType, args...)
        @test convert(RetType, res) == res_converted
        @test res_converted == pycall_res
        RetType != PyAny && @test res_converted isa RetType
    end

    @testset "basics" begin
        args = ("a", 2, 4.5)
        for RetType in (PyObject, PyAny, Tuple)
            pycall_checks(res, pymklist, RetType, args...)
            gc()
        end
    end
end
