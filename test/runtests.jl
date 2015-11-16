using Base.Test, PyCall, Compat

PYTHONPATH=get(ENV,"PYTHONPATH","")
PYTHONHOME=get(ENV,"PYTHONHOME","")
PYTHONEXECUTABLE=get(ENV,"PYTHONEXECUTABLE","")
info("Python version $pyversion from $(PyCall.libpython), PYTHONHOME=$(PyCall.PYTHONHOME)\nENV[PYTHONPATH]=$PYTHONPATH\nENV[PYTHONHOME]=$PYTHONHOME\nENV[PYTHONEXECUTABLE]=$PYTHONEXECUTABLE")

roundtrip(T, x) = convert(T, PyObject(x))
roundtrip(x) = roundtrip(PyAny, x)
roundtripeq(T, x) = roundtrip(T, x) == x
roundtripeq(x) = roundtrip(x) == x


# test handling of type-tuple changes in Julia 0.4
import PyCall.pyany_toany
@test pyany_toany(Int) == Int
@test pyany_toany(PyAny) == Any
@test pyany_toany(@compat Tuple{Int,PyAny}) == @compat Tuple{Int,Any}
@test pyany_toany(@compat Tuple{Int,Tuple{PyAny,Int8}}) == @compat Tuple{Int,Tuple{Any,Int8}}
if VERSION >= v"0.4.0-dev+4319"
    @test pyany_toany(@compat Tuple{PyAny,Int,Vararg{PyAny}}) == @compat Tuple{Any,Int,Vararg{Any}}
end

@test roundtripeq(17)
@test roundtripeq(0x39)
@test roundtripeq(true) && roundtripeq(false)
@test roundtripeq(3.14159)
@test roundtripeq(1.3+4.5im)
@test roundtripeq(nothing)
@test roundtripeq("Hello world")
@test roundtripeq("Hëllö")
@test roundtripeq("Hello \0\0\0world")
@test roundtripeq("Hël\0\0lö")
@test roundtripeq(Symbol, :Hello)
@test roundtripeq(C_NULL) && roundtripeq(convert(Ptr{Void}, 12345))
@test roundtripeq([1,3,4,5]) && roundtripeq([1,3.2,"hello",true])
@test roundtripeq([1 2 3;4 5 6]) && roundtripeq([1. 2 3;4 5 6])
@test roundtripeq((1,(3.2,"hello"),true)) && roundtripeq(())
@test roundtripeq(Int32)
@test roundtripeq(@compat Dict(1 => "hello", 2 => "goodbye")) && roundtripeq(Dict())
@test roundtripeq(UInt8[1,3,4,5])
@test convert(Vector{UInt8}, "hello") == "hello".data

@test pycall(PyObject(x -> x + 1), PyAny, 314158) == 314159
if VERSION >= v"0.4.0-dev+1246" # call overloading
    @test PyObject(x -> x + 1)(314158) == 314159
    @test PyAny(PyObject(3)) == 3
end
@test roundtrip(x -> x + 1)(314158) == 314159

testkw(x; y=0) = x + 2*y
@test pycall(PyObject(testkw), PyAny, 314157) == 314157
@test pycall(PyObject(testkw), PyAny, 314157, y=1) == 314159
@test roundtrip(testkw)(314157) == 314157
@test roundtrip(testkw)(314157, y=1) == 314159

if PyCall.npy_initialized
    @test PyArray(PyObject([1. 2 3;4 5 6])) == [1. 2 3;4 5 6]
end
@test PyVector(PyObject([1,3.2,"hello",true])) == [1,3.2,"hello",true]
@test PyDict(PyObject(@compat Dict(1 => "hello", 2 => "goodbye"))) == @compat Dict(1 => "hello", 2 => "goodbye")

let d = PyDict(@compat Dict(1 => "hello", 34 => "yes" ))
    @test get(d.o, 1) == "hello"
    set!(d.o, 34, "goodbye")
    @test d[34] == "goodbye"
    @test sort!(keys(Int, d)) == sort!(collect(d.o[:keys]())) == sort!(collect(keys(d))) == [1, 34]
end

@test roundtripeq(Any[1 2 3; 4 5 6])
@test roundtripeq([])
@test convert(Array{PyAny,1}, PyObject(Any[1 2 3; 4 5 6])) == Any[Any[1,2,3],Any[4,5,6]]
if PyCall.npy_initialized
    @test roundtripeq(begin A = Array(Int); A[1] = 3; A; end)
end
@test convert(PyAny, PyObject(begin A = Array(Any); A[1] = 3; A; end)) == 3

array2py2arrayeq(x) = PyCall.py2array(Float64,PyCall.array2py(x)) == x
@test array2py2arrayeq(rand(3))
@test array2py2arrayeq(rand(3,4))
@test array2py2arrayeq(rand(3,4,5))

@test roundtripeq(2:10) && roundtripeq(10:-1:2)
@test roundtrip(2:2.0:10) == convert(Vector{Float64}, 2:2.0:10)

@pyimport math
@test_approx_eq math.sin(3) sin(3)

@test collect(PyObject([1,"hello",5])) == [1,"hello",5]

@test try @eval (@pyimport os.path) catch ex isa(ex, ArgumentError) end

@test PyObject("hello") == PyObject("hello")
@test PyObject("hello") != PyObject("hellö")
@test PyObject(hash) == PyObject(hash)
@test PyObject(hash) != PyObject(println)
@test hash(PyObject("hello")) == hash(PyObject("hello"))
@test hash(PyObject("hello")) != hash(PyObject("hellö"))
@test hash(PyObject("hello")) != hash("hellö")
@test hash(PyObject(hash)) == hash(PyObject(hash))
@test hash(PyObject(hash)) != hash(PyObject(println))
@test hash(PyObject(hash)) != hash(hash)

# issue #92:
let x = PyVector(PyAny[])
    pyeval("lambda x: x.append(\"bar\")")(x)
    @test x == ["bar"]
end

if pyversion >= v"2.7" && isdefined(PyCall, :PyDateTime_CAPI)
    # Dates is built-in in Julia 0.4
    if !isdefined(Base, :Dates)
        import Dates
    end
    @test roundtripeq(Dates.Date(2012,3,4))
    @test roundtripeq(Dates.DateTime(2012,3,4, 7,8,9,11))
    @test roundtripeq(Dates.Millisecond(typemax(Int32)))
    @test roundtripeq(Dates.Millisecond(typemin(Int32)))
    @test roundtripeq(Dates.Second, Dates.Second(typemax(Int32)))
    @test roundtripeq(Dates.Second, Dates.Second(typemin(Int32)))
    @test roundtripeq(Dates.Day, Dates.Day(999999999)) # max allowed day timedelta
    @test roundtripeq(Dates.Day, Dates.Day(-999999999)) # min allowed day timedelta
end

# fixme: is there any nontrivial mimewritable test we can do?
@test !mimewritable("text/html", PyObject(1))

# in Python 3, we need a specific encoding to write strings or bufferize them
# (http://stackoverflow.com/questions/5471158/typeerror-str-does-not-support-the-buffer-interface)
pyutf8(s::PyObject) = pycall(s["encode"], PyObject, "utf-8")
pyutf8(s::ByteString) = pyutf8(PyObject(s))

# IO (issue #107)
@test roundtripeq(STDOUT)
let buf = IOBuffer(false, true), obuf = PyObject(buf)
    @test !obuf[:isatty]()
    @test obuf[:writable]()
    @test !obuf[:readable]()
    @test obuf[:seekable]()
    obuf[:write](pyutf8("hello"))
    @test position(buf) == obuf[:tell]() == 5
    let p = obuf[:seek](-2, 1)
        @test p == position(buf) == 3
    end
    let p = obuf[:seek](0, 0)
        @test p == position(buf) == 0
    end
    @test takebuf_string(buf) == "hello"
    obuf[:writelines](["first\n", "second\n", "third"])
    @test takebuf_string(buf) == "first\nsecond\nthird"
    obuf[:write](convert(Vector{UInt8}, "möre stuff"))
    @test takebuf_string(buf) == "möre stuff"
    @test isopen(buf) == !obuf[:closed] == true
    obuf[:close]()
    @test isopen(buf) == !obuf[:closed] == false
end
let buf = IOBuffer("hello\nagain"), obuf = PyObject(buf)
    @test !obuf[:writable]()
    @test obuf[:readable]()
    @test obuf[:readlines]() == ["hello\n","again"]
end
let buf = IOBuffer("hello\nagain"), obuf = PyObject(buf)
    @test obuf[:readall]() == convert(Vector{UInt8}, "hello\nagain")
end
let buf = IOBuffer("hello\nagain"), obuf = PyTextIO(buf)
    @test obuf[:encoding] == "UTF-8"
    @test obuf[:readall]() == "hello\nagain"
end
let nm = tempname()
    open(nm, "w") do f
        @test roundtripeq(f)
        pf = PyObject(f)
        @test pf[:fileno]() == fd(f)
        @test pf[:writable]()
        @test !pf[:readable]()
        pf[:write](pyutf8(nm))
        pf[:flush]()
    end
    @test readall(nm) == nm
end

# issue #112
@test roundtripeq(Array, [1,2,3,4])
@test roundtripeq(Array{Int8}, [1,2,3,4])

# conversion of numpy scalars
pyanycheck(x::Any) = pyanycheck(typeof(x), PyObject(x))
pyanycheck(T, o::PyObject) = isa(convert(PyAny, o), T)
@test pyanycheck(Int, PyVector{PyObject}(PyObject([1]))[1])
@test pyanycheck(Float64, PyVector{PyObject}(PyObject([1.3]))[1])
@test pyanycheck(Complex128, PyVector{PyObject}(PyObject([1.3+1im]))[1])

pymodule_exists(s::AbstractString) = try
    pyimport(s)
    true
catch
    false
end

# bigfloat conversion
if pymodule_exists("mpmath")
    for x in (big(pi), big(pi) + im/big(pi))
        @test pyanycheck(x)
        # conversion may not be exact since it goes through a decimal string
        @test abs(roundtrip(x) - x) < eps(BigFloat) * 1e3 * abs(x)
    end
end
@test convert(BigInt, PyObject(1234)) == 1234

# buffers
let b = PyCall.PyBuffer(pyutf8("test string"))
    @test ndims(b) == 1
    @test (length(b),) == (length("test string"),) == (size(b, 1),) == size(b)
    @test stride(b, 1) == 1
    @test PyCall.iscontiguous(b) == true
end

let o = PyObject(1+2im)
    @test haskey(o, :real)
    @test :real in keys(o)
    @test o[:real] == 1
end

# []-based sequence access
let a1=[5,8,6], a2=rand(3,4), a3=rand(3,4,5), o1=PyObject(a1), o2=PyObject(a2), o3=PyObject(a3)
    @test [o1[i] for i in eachindex(a1)] == a1
    @test [o1[end-(i-1)] for i in eachindex(a1)] == reverse(a1)
    @test o2[1] == collect(a2[1,:])
    @test length(o1) == length(o2) == length(o3) == 3
    o1[end-1] = 7
    @test o1[2] == 7

    # multiple indices are passed as tuples, but this is apparently
    # only supported by numpy arrays.
    if PyCall.npy_initialized
        @test [o2[i,j] for i=1:3, j=1:4] == a2
        @test [o3[i,j,k] for i=1:3, j=1:4, k=1:5] == a3
        @test o3[2,3] == collect(a3[2,3,:])
        o2[2,3] = 8
        @test o2[2,3] == 8
        o3[2,3,4] = 9
        @test o3[2,3,4] == 9
    end
end

# list operations:
let o = PyObject(Any[8,3])
    @test collect(push!(o, 5)) == [8,3,5]
    @test pop!(o) == 5 && collect(o) == [8,3]
    @test shift!(o) == 8 && collect(o) == [3]
    @test collect(unshift!(o, 9)) == [9,3]
    @test collect(prepend!(o, [5,4,2])) == [5,4,2,9,3]
    @test collect(append!(o, [1,6,8])) == [5,4,2,9,3,1,6,8]
    @test isempty(empty!(o))
end

# issue #216:
@test length(collect(pyimport("itertools")[:combinations]([1,2,3],2))) == 3
