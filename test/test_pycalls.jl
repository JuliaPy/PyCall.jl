using Compat.Test, PyCall

@testset "pywrapfn" begin
    np = pyimport("numpy")
    ops = pyimport("operator")
    eq = ops["eq"]
    npzeros = np["zeros"]
    npzeros_pyo(sz, dtype="d", order="F")     = pycall(npzeros, PyObject, sz, dtype, order)
    npzeros_pyany(sz, dtype="d", order="F")   = pycall(npzeros, PyAny, sz, dtype, order)
    npzeros_pyarray(sz, dtype="d", order="F") = pycall(npzeros, PyArray, sz, dtype, order)

    npzeros2dwrap_pyo     = pywrapfn(npzeros, 3) # PyObject is default returntype
    npzeros2dwrap_pyany   = pywrapfn(npzeros, 3, PyAny)
    npzeros2dwrap_pyarray = pywrapfn(npzeros, 3, PyArray)

    arr_size = (2,2)

    # all args
    @test np["array_equal"](npzeros2dwrap_pyo((arr_size, "d", "C")), npzeros_pyo(arr_size))
    # args already set
    @test np["array_equal"](npzeros2dwrap_pyo(), npzeros_pyo(arr_size))

    @test all(npzeros2dwrap_pyany((arr_size, "d", "C")) .== npzeros_pyany(arr_size))
    @test all(npzeros2dwrap_pyany() .== npzeros_pyany(arr_size))

    @test all(npzeros2dwrap_pyarray((arr_size, "d", "C")) .== npzeros_pyarray(arr_size))
    @test all(npzeros2dwrap_pyarray() .== npzeros_pyarray(arr_size))

    @testset "setarg(s)!" begin
        arr_size = (3,3)
        # set arg 1, then call without args, old args should be unchanged
        setarg!(npzeros2dwrap_pyo, arr_size, 1)
        int32mat_pyo = npzeros2dwrap_pyo()
        @test np["array_equal"](int32mat_pyo, npzeros_pyo(arr_size))
        @test int32mat_pyo["dtype"] == np["dtype"]("d")
        @test int32mat_pyo["flags"]["c_contiguous"] == true
        @test int32mat_pyo["shape"] == arr_size

        # set arg 2 (data type), then call without args
        setarg!(npzeros2dwrap_pyo, "i", 2)
        int32mat_pyo1 = npzeros2dwrap_pyo()
        @test int32mat_pyo1["dtype"] == np["dtype"]("i")
        @test int32mat_pyo1["flags"]["c_contiguous"] == true
        @test int32mat_pyo1["shape"] == arr_size

        # set arg 3 (order - C or Fortran), then call without args
        setarg!(npzeros2dwrap_pyo, "F", 3)
        int32mat_pyo2 = npzeros2dwrap_pyo()
        @test int32mat_pyo2["flags"]["f_contiguous"] == true
        @test int32mat_pyo2["dtype"] == np["dtype"]("i")
        @test int32mat_pyo2["shape"] == arr_size

        # set all args then call without args
        arr_size = (4,4)
        setargs!(npzeros2dwrap_pyo, (arr_size, "c", "F"), 3)
        cmplxmat_pyo = npzeros2dwrap_pyo()
        @test cmplxmat_pyo["dtype"] == np["dtype"]("c")
        @test cmplxmat_pyo["flags"]["f_contiguous"] == true
        @test cmplxmat_pyo["shape"] == arr_size
    end
end
