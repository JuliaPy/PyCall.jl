using PyCall, BenchmarkTools, DataStructures

results = OrderedDict{String,Any}()

let
    np = pyimport("numpy")
    nprand = np["random"]["rand"]
    nprand_pyo(sz...) = pycall(nprand, PyObject, sz...)
    nprand_pyo!(ret::PyObject, sz...) = pycall!(ret, nprand, PyObject, sz...)
    ret = PyNULL()
    args_lens = (0,3,7,12,17)
    arr_sizes = (ntuple(i->1, len) for len in args_lens)
    nprand_wraps = [PyFuncWrap(nprand, map(typeof, arr_size)) for arr_size in arr_sizes]
    @show typeof(nprand_wraps)
    for (i, arr_size) in enumerate(arr_sizes)
        nprand_wrap = nprand_wraps[i]
        arr_size_str = args_lens[i] < 5 ? "$arr_size" : "$(args_lens[i])*(1,1,...)"
        results["nprand_pyo $arr_size_str"] = @benchmark $nprand_pyo($arr_size...)
        println("nprand_pyo $arr_size_str:\n"); display(results["nprand_pyo $arr_size_str"])
        println("--------------------------------------------------")

        results["nprand_pyo! $arr_size_str"] = @benchmark $nprand_pyo!($ret, $arr_size...)
        println("nprand_pyo! $arr_size_str:\n"); display(results["nprand_pyo! $arr_size_str"])
        println("--------------------------------------------------")

        results["nprand_wrap $arr_size_str"] = @benchmark $nprand_wrap($arr_size...)
        println("nprand_wrap $arr_size_str:\n"); display(results["nprand_wrap $arr_size_str"])
        println("--------------------------------------------------")

        # args already set by nprand_wrap calls above
        results["nprand_wrap_noargs $arr_size_str"] = @benchmark $nprand_wrap()
        println("nprand_wrap_noargs $arr_size_str:\n"); display(results["nprand_wrap_noargs $arr_size_str"])
        println("--------------------------------------------------")
    end
end

println("")
println("Mean times")
println("----------")
foreach((r)->println(rpad(r[1],33), ": ", mean(r[2])), results)
