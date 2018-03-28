using Compat.Test, PyCall

@testset "PyFuncWrap" begin
    np = pyimport("numpy")
    ops = pyimport("operator")
    eq = ops["eq"]
    npzeros = np["zeros"]
    npzeros_pyo(sz, dtype="d", order="F")     = pycall(npzeros, PyObject, sz, dtype, order)
    npzeros_pyany(sz, dtype="d", order="F")   = pycall(npzeros, PyAny, sz, dtype, order)
    npzeros_pyarray(sz, dtype="d", order="F") = pycall(npzeros, PyArray, sz, dtype, order)

    # PyObject is default returntype
    npzeros2dwrap_pyo     = PyFuncWrap(npzeros, ((Int, Int), String, String))
    npzeros2dwrap_pyany   = PyFuncWrap(npzeros, ((Int, Int), String, String), PyAny)
    npzeros2dwrap_pyarray = PyFuncWrap(npzeros, ((Int, Int), String, String), PyArray)

    arr_size = (2,2)

    # all args
    @test np["array_equal"](npzeros2dwrap_pyo(arr_size, "d", "F"), npzeros_pyo(arr_size))
    # args already set
    @test np["array_equal"](npzeros2dwrap_pyo(), npzeros_pyo(arr_size))

    @test all(npzeros2dwrap_pyany(arr_size, "d", "F") .== npzeros_pyany(arr_size))
    @test all(npzeros2dwrap_pyany() .== npzeros_pyany(arr_size))

    @test all(npzeros2dwrap_pyarray(arr_size, "d", "F") .== npzeros_pyarray(arr_size))
    @test all(npzeros2dwrap_pyarray() .== npzeros_pyarray(arr_size))
end
