using PyCall, BenchmarkTools, DataStructures

results = OrderedDict{String,Any}()

let
    np = pyimport("numpy")
    nprand = np["random"]["rand"]
    nprand_pyo(sz...) = pycall(nprand, PyObject, sz...)
    nprand2d_wrap = PyFuncWrap(nprand, (Int, Int))

    arr_size = (2,2)

    results["nprand_pyo"] = @benchmark $nprand_pyo($arr_size...)
    println("nprand_pyo:\n"); display(results["nprand_pyo"])
    println("--------------------------------------------------")

    results["nprand2d_wrap"] = @benchmark $nprand2d_wrap($arr_size...)
    println("nprand2d_wrap:\n"); display(results["nprand2d_wrap"])
    println("--------------------------------------------------")

    # args already set by nprand2d_wrap calls above
    results["nprand2d_wrap_noargs"] = @benchmark $nprand2d_wrap()
    println("nprand2d_wrap_noargs:\n"); display(results["nprand2d_wrap_noargs"])
    println("--------------------------------------------------")

    arr_size = ntuple(i->2, 15)

    results["nprand_pyo2"] = @benchmark $nprand_pyo($arr_size...)
    println("nprand_pyo2:\n"); display(results["nprand_pyo2"])
    println("--------------------------------------------------")

    results["nprand2d_wrap2"] = @benchmark $nprand2d_wrap($arr_size...)
    println("nprand2d_wrap2:\n"); display(results["nprand2d_wrap2"])
    println("--------------------------------------------------")

    # args already set by nprand2d_wrap calls above
    results["nprand2d_wrap_noargs2"] = @benchmark $nprand2d_wrap()
    println("nprand2d_wrap_noargs2:\n"); display(results["nprand2d_wrap_noargs2"])
    println("--------------------------------------------------")
end

println("")
println("Mean times")
println("----------")
foreach((r)->println(rpad(r[1],23), ": ", mean(r[2])), results)
