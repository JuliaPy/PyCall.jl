using Base.Test
using PyCall

roundtrip(T, x) = convert(T, PyObject(x))
roundtrip(x) = roundtrip(PyAny, x)
roundtripeq(T, x) = roundtrip(T, x) == x
roundtripeq(x) = roundtrip(x) == x

@test roundtripeq(17)
@test roundtripeq(0x39)
@test roundtripeq(true) && roundtripeq(false)
@test roundtripeq(3.14159)
@test roundtripeq(1.3+4.5im)
@test roundtripeq(nothing)
@test roundtripeq("Hello world")
@test roundtripeq("Hëllö")
@test roundtripeq(Symbol, :Hello)
@test roundtripeq(C_NULL) && roundtripeq(convert(Ptr{Void}, 12345))
@test roundtripeq([1,3,4,5]) && roundtripeq([1,3.2,"hello",true])
@test roundtripeq([1 2 3;4 5 6]) && roundtripeq([1. 2 3;4 5 6])
@test roundtripeq((1,(3.2,"hello"),true)) && roundtripeq(())
@test roundtripeq(Int32)
@test roundtripeq([1 => "hello", 2 => "goodbye"]) && roundtripeq(Dict())
@test roundtripeq(Uint8[1,3,4,5])

@test pycall(PyObject(x -> x + 1), PyAny, 314158) == 314159
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
@test PyDict(PyObject([1 => "hello", 2 => "goodbye"])) == [1 => "hello", 2 => "goodbye"]

let d = PyDict([1 => "hello", "yes" => 34])
    @test get(d.o, 1) == "hello"
    set!(d.o, "yes", "goodbye")
    @test d["yes"] == "goodbye"
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
@test roundtrip(2:2.0:10) == [2:2.0:10]

@pyimport math
@test_approx_eq math.sin(3) sin(3)

@test collect(PyObject([1,"hello",5])) == [1,"hello",5]

@test try @eval (@pyimport os.path) catch ex isa(ex, ArgumentError) end

@test PyObject("hello") == PyObject("hello")
@test PyObject("hello") != PyObject("hellö")
@test PyObject(hash) == PyObject(hash)
@test PyObject(hash) != PyObject(bitmix)
@test hash(PyObject("hello")) == hash(PyObject("hello"))
@test hash(PyObject("hello")) != hash(PyObject("hellö"))
@test hash(PyObject("hello")) != hash("hellö")
@test hash(PyObject(hash)) == hash(PyObject(hash))
@test hash(PyObject(hash)) != hash(PyObject(bitmix))
@test hash(PyObject(hash)) != hash(hash)

# issue #92:
let x = PyVector(PyAny[])
    pyeval("lambda x: x.append(\"bar\")")(x)
    @test x == ["bar"]
end

if pyversion >= v"2.7"
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

# IO (issue #107)
@test roundtripeq(STDOUT)
let buf = IOBuffer(false, true), obuf = PyObject(buf)
    @test !obuf[:isatty]()
    @test obuf[:writable]()
    @test !obuf[:readable]()
    @test obuf[:seekable]()
    obuf[:write]("hello")
    @test position(buf) == obuf[:tell]() == 5
    @test obuf[:seek](-2, 1) == position(buf) == 3
    @test obuf[:seek](0, 0) == position(buf) == 0
    @test takebuf_string(buf) == "hello"
    obuf[:writelines](["first\n", "second\n", "third"])
    @test takebuf_string(buf) == "first\nsecond\nthird"
    obuf[:write](convert(Vector{Uint8}, "möre stuff"))
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
    @test obuf[:readall]() == convert(Vector{Uint8}, "hello\nagain")
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
        pf[:write](nm)
        pf[:flush]()
    end
    @test readall(nm) == nm
end
