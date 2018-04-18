using Compat.Test, PyCall

@testset "pycall!" begin
    np = pyimport("numpy")
    ops = pyimport("operator")
    eq = ops["eq"]
    npzeros = np["zeros"]
    res = PyNULL()
    npzeros_pyo(sz, dtype="d", order="F")     = pycall(npzeros, PyObject, sz, dtype, order)
    npzeros_pyany(sz, dtype="d", order="F")   = pycall(npzeros, PyAny, sz, dtype, order)
    npzeros_pyarray(sz, dtype="d", order="F") = pycall(npzeros, PyArray, sz, dtype, order)

    npzeros_pyo!(ret, sz, dtype="d", order="F")     = pycall!(ret, npzeros, PyObject, sz, dtype, order)
    npzeros_pyany!(ret, sz, dtype="d", order="F")   = pycall!(ret, npzeros, PyAny, sz, dtype, order)
    npzeros_pyarray!(ret, sz, dtype="d", order="F") = pycall!(ret, npzeros, PyArray, sz, dtype, order)

    arr_size = (3, 4)

    @testset "basics" begin
        @test np["array_equal"](npzeros_pyo!(res, arr_size), npzeros_pyo(arr_size))
        @test all(npzeros_pyany!(res, arr_size) .== npzeros_pyany(arr_size))
        @test all(npzeros_pyarray!(res, arr_size) .== npzeros_pyarray(arr_size))
        @test npzeros_pyany!(res, arr_size) isa Array
        @test npzeros_pyarray!(res, arr_size) isa PyArray
    end
end
