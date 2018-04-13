using PyCall, BenchmarkTools, DataStructures
using PyCall: _pycall!, pycall_legacy

results = OrderedDict{String,Any}()

let
    np = pyimport("numpy")
    nprand = np["random"]["rand"]
    ret = PyNULL()
    args_lens = (0,1,2,3,7,12,17)
    # args_lens = (1,3,7)
    arr_sizes = (ntuple(i->1, len) for len in args_lens)

    for (i, arr_size) in enumerate(arr_sizes)
        nprand_wrap = pywrapfn(nprand, length(arr_size))
        pyargsptr = ccall((@pysym :PyTuple_New), PyPtr, (Int,), length(arr_size))
        arr_size_str = args_lens[i] < 5 ? "$arr_size" : "$(args_lens[i])*(1,1,...)"

        results["pycall_legacy $arr_size_str"] = @benchmark pycall_legacy($nprand, PyObject, $arr_size...)
        println("pycall_legacy $arr_size_str:\n"); display(results["pycall_legacy $arr_size_str"])
        println("--------------------------------------------------")

        results["pycall $arr_size_str"] = @benchmark pycall($nprand, PyObject, $arr_size...)
        println("pycall $arr_size_str:\n"); display(results["pycall $arr_size_str"])
        println("--------------------------------------------------")

        results["pycall! $arr_size_str"] = @benchmark pycall!($ret, $nprand, PyObject, $arr_size...)
        println("pycall! $arr_size_str:\n"); display(results["pycall! $arr_size_str"])
        println("--------------------------------------------------")

        results["_pycall! $arr_size_str"] = @benchmark $_pycall!($ret, $pyargsptr, $nprand, $arr_size)
        println("_pycall! $arr_size_str:\n"); display(results["_pycall! $arr_size_str"])
        println("--------------------------------------------------")

        results["nprand_wrap $arr_size_str"] = @benchmark $nprand_wrap($arr_size)
        println("nprand_wrap $arr_size_str:\n"); display(results["nprand_wrap $arr_size_str"])
        println("--------------------------------------------------")

        # args already set by nprand_wrap calls above
        results["nprand_wrap_noargs $arr_size_str"] = @benchmark $nprand_wrap()
        println("nprand_wrap_noargs $arr_size_str:\n"); display(results["nprand_wrap_noargs $arr_size_str"])
        println("--------------------------------------------------")
    end
end
#
println("")
println("Mean times")
println("----------")
foreach((r)->println(rpad(r[1],33), "\t", mean(r[2])), results)
println("")
println("Median times")
println("----------")
foreach((r)->println(rpad(r[1],33), "\t", median(r[2])), results)
