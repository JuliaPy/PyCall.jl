using PyCall, BenchmarkTools, DataStructures
using PyCall: PyArray_Info

results = OrderedDict{String,Any}()

let
    np = pyimport("numpy")
    nprand = np["random"]["rand"]
    # nparray_pyo(x) = pycall(np["array"], PyObject, x)
    # pytestarray(sz::Int...) = pycall(np["reshape"], PyObject, nparray_pyo(1:prod(sz)), sz)

    # no convert baseline
    nprand_pyo(sz...)   = pycall(nprand, PyObject, sz...)

    for arr_size in [(2,2), (100,100)]
        pyo_arr = nprand_pyo(arr_size...)
        results["nprand_pyo$arr_size"] = @benchmark $nprand_pyo($arr_size...)
        println("nprand_pyo $arr_size:\n"); display(results["nprand_pyo$arr_size"])
        println("--------------------------------------------------")

        results["convert_pyarr$arr_size"] = @benchmark $convert(PyArray, $pyo_arr)
        println("convert_pyarr $arr_size:\n"); display(results["convert_pyarr$arr_size"])
        println("--------------------------------------------------")

        results["PyArray_Info$arr_size"] = @benchmark $PyArray_Info($pyo_arr)
        println("PyArray_Info $arr_size:\n"); display(results["PyArray_Info$arr_size"])
        println("--------------------------------------------------")

        results["convert_pyarrbuf$arr_size"] = @benchmark $PyArray($pyo_arr)
        println("convert_pyarrbuf $arr_size:\n"); display(results["convert_pyarrbuf$arr_size"])
        println("--------------------------------------------------")

        results["convert_arr$arr_size"] = @benchmark convert(Array, $pyo_arr)
        println("convert_arr $arr_size:\n"); display(results["convert_arr$arr_size"])
        println("--------------------------------------------------")

        results["convert_arrbuf$arr_size"] = @benchmark $NoCopyArray($pyo_arr)
        println("convert_arrbuf $arr_size:\n"); display(results["convert_arrbuf$arr_size"])
        println("--------------------------------------------------")

        pyarr = convert(PyArray, pyo_arr)
        results["setdata!$arr_size"] = @benchmark $setdata!($pyarr, $pyo_arr)
        println("setdata!:\n"); display(results["setdata!$arr_size"])
        println("--------------------------------------------------")

        pyarr = convert(PyArray, pyo_arr)
        pybuf=PyBuffer()
        results["setdata! bufprealloc$arr_size"] =
            @benchmark $setdata!($pyarr, $pyo_arr, $pybuf)
        println("setdata! bufprealloc:\n"); display(results["setdata! bufprealloc$arr_size"])
        println("--------------------------------------------------")
    end
end
println()
println("Mean times")
println("----------")
foreach((r)->println(rpad(r[1],27), ": ", mean(r[2])), results)
