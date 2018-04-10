using PyCall, BenchmarkTools, DataStructures

results = OrderedDict{String,Any}()
let
    np = pyimport("numpy")
    nprandint = np["random"]["randint"]
    nprand = np["random"]["rand"]
    res = PyNULL()

    tuplen = 16
    tpl = convert(PyObject, (1:tuplen...))
    lst = convert(PyObject, Any[1:tuplen...])
    for (name, pycoll) in zip(("tpl", "lst"), (tpl, lst))
        idx = rand(0:(tuplen-1))
        results["standard get $name"] = @benchmark get($pycoll, PyObject, PyObject($idx))
        println("standard get:\n"); display(results["standard get $name"])
        println("--------------------------------------------------")

        idx = rand(0:(tuplen-1))
        results["faster get $name"] = @benchmark get($pycoll, PyObject, $idx)
        println("faster get:\n"); display(results["faster get $name"])
        println("--------------------------------------------------")

        idx = rand(0:(tuplen-1))
        results["get! $name"] = @benchmark get!($res, $pycoll, PyObject, $idx)
        println("get!:\n"); display(results["get! $name"])
        println("--------------------------------------------------")

        if pycoll == tpl
            idx = rand(0:(tuplen-1))
            results["unsafe_gettpl!"] = @benchmark unsafe_gettpl!($res, $tpl, PyObject, $idx)
            println("unsafe_gettpl!:\n"); display(results["unsafe_gettpl!"])
            println("--------------------------------------------------")
        end
    end
end

println("")
println("Mean times")
println("----------")
foreach((r)->println(rpad(r[1],20), ": ", mean(r[2])), results)
