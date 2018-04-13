using PyCall, BenchmarkTools, DataStructures
using PyCall: _pycall!

results = OrderedDict{String,Any}()
@inline nprand_pyo(o::PyObject, sz...) = pycall(o, PyObject, sz...)
# nprand_pyo!(ret::PyObject, sz...) = pycall!(ret, nprand, PyObject, sz...)
# _nprand_pyo!(ret::PyObject, sz...) = _pycall!(ret, nprand, sz...)
@inline nprand_pyo!(ret::PyObject, o::PyObject, sz...) = pycall!(ret, o, PyObject, sz...)
@inline _nprand_pyo!(ret::PyObject, o::PyObject, pyargsptr::PyPtr, sz...) =
    _pycall!(ret, pyargsptr, o, sz)


function fastercall(o::PyObject, nargs::Int)
    pyargsptr = ccall((@pysym :PyTuple_New), PyPtr, (Int,), nargs)
    ret = PyNULL()
    (args) -> _pycall!(ret, pyargsptr, o, args, nargs, C_NULL)
end

let
    np = pyimport("numpy")
    nprand = np["random"]["rand"]
    ret = PyNULL()
    # args_lens = (0,3,7,12,17)
    # args_lens = (1,3,7)
    nargs = 7
    args_lens = (nargs,)
    arr_sizes = (ntuple(i->1, len) for len in args_lens)
    nprand_wraps = [PyFuncWrap(nprand, map(typeof, arr_size)) for arr_size in arr_sizes]
    pyargsptr = ccall((@pysym :PyTuple_New), PyPtr, (Int,), nargs)

    # oargs = Array{PyObject}(7)
    nprand_pyo!(ret, nprand, first(arr_sizes)...)
    _nprand_pyo!(ret, nprand, pyargsptr, first(arr_sizes)...)
    # Profile.clear_malloc_data()
    # @code_warntype _pycall!(ret, nprand, first(arr_sizes)...)
    for (i, arr_size) in enumerate(arr_sizes)
        nprand_wrap = nprand_wraps[i]
        fastrand = fastercall(nprand, length(arr_size))
        arr_size_str = args_lens[i] < 5 ? "$arr_size" : "$(args_lens[i])*(1,1,...)"

        results["nprand_wrap $arr_size_str"] = @benchmark $nprand_wrap($arr_size)
        println("nprand_wrap $arr_size_str:\n"); display(results["nprand_wrap $arr_size_str"])
        println("--------------------------------------------------")

        # args already set by nprand_wrap calls above
        results["nprand_wrap_noargs $arr_size_str"] = @benchmark $nprand_wrap()
        println("nprand_wrap_noargs $arr_size_str:\n"); display(results["nprand_wrap_noargs $arr_size_str"])
        println("--------------------------------------------------")

        # results["nprand_pyo $arr_size_str"] = @benchmark $nprand_pyo($arr_size...)
        # println("nprand_pyo $arr_size_str:\n"); display(results["nprand_pyo $arr_size_str"])
        # println("--------------------------------------------------")

        results["nprand_pyo! $arr_size_str"] = @benchmark $nprand_pyo!($ret, $nprand, $arr_size...)
        println("nprand_pyo! $arr_size_str:\n"); display(results["nprand_pyo! $arr_size_str"])
        println("--------------------------------------------------")

        results["fastrand $arr_size_str"] = @benchmark $fastrand($arr_size)
        println("fastrand $arr_size_str:\n"); display(results["fastrand $arr_size_str"])
        println("--------------------------------------------------")

        results["_nprand_pyo! $arr_size_str"] = @benchmark $_nprand_pyo!($ret, $nprand, $pyargsptr, $arr_size...)
        println("_nprand_pyo! $arr_size_str:\n"); display(results["_nprand_pyo! $arr_size_str"])
        println("--------------------------------------------------")

        results["pycall! $arr_size_str"] = @benchmark pycall!($ret, $nprand, PyObject, $arr_size...)
        println("pycall! $arr_size_str:\n"); display(results["pycall! $arr_size_str"])
        println("--------------------------------------------------")

        results["_pycall! $arr_size_str"] = @benchmark $_pycall!($ret, $pyargsptr, $nprand, $arr_size)
        println("_pycall! $arr_size_str:\n"); display(results["_pycall! $arr_size_str"])
        println("--------------------------------------------------")
        # _nprand_pyo!(ret, pyargsptr, oargs, arr_size...)
    end
end
#
println("")
println("Mean times")
println("----------")
foreach((r)->println(rpad(r[1],33), ": ", mean(r[2])), results)
println("")
println("Median times")
println("----------")
foreach((r)->println(rpad(r[1],33), ": ", median(r[2])), results)
