using PyCall, BenchmarkTools, DataStructures

results = OrderedDict{String,Any}()
let
    np = pyimport("numpy")
    nprandint = np["random"]["randint"]
    nprand = np["random"]["rand"]
    res = PyNULL()

    tuplen = 16
    tpl = convert(PyObject, (1:tuplen...))

    tplidx = rand(0:(tuplen-1))
    results["standard get"] = @benchmark get($tpl, PyObject, $tplidx)
    println("standard get:\n"); display(results["standard get"])
    println("--------------------------------------------------")

    tplidx = rand(0:(tuplen-1))
    results["get!"] = @benchmark get!($res, $tpl, PyObject, $tplidx)
    println("get!:\n"); display(results["get!"])
    println("--------------------------------------------------")

    tplidx = rand(0:(tuplen-1))
    results["unsafe_gettpl!"] = @benchmark unsafe_gettpl!($res, $tpl, PyObject, $tplidx)
    println("unsafe_gettpl!:\n"); display(results["unsafe_gettpl!"])
    println("--------------------------------------------------")
end

println("")
println("Mean times")
println("----------")
foreach((r)->println(rpad(r[1],20), ": ", mean(r[2])), results)
