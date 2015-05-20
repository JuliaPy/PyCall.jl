using Base.Test, PyCall, Compat

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
@test roundtripeq(Symbol, :Hello)
@test roundtripeq(C_NULL) && roundtripeq(convert(Ptr{Void}, 12345))
@test roundtripeq([1,3,4,5]) && roundtripeq([1,3.2,"hello",true])
@test roundtripeq([1 2 3;4 5 6]) && roundtripeq([1. 2 3;4 5 6])
@test roundtripeq((1,(3.2,"hello"),true)) && roundtripeq(())
@test roundtripeq(Int32)
@test roundtripeq(@compat Dict(1 => "hello", 2 => "goodbye")) && roundtripeq(Dict())
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
@test PyDict(PyObject(@compat Dict(1 => "hello", 2 => "goodbye"))) == @compat Dict(1 => "hello", 2 => "goodbye")

let d = PyDict(@compat Dict(1 => "hello", "yes" => 34))
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
        pf[:write](pyutf8(nm))
        pf[:flush]()
    end
    @test readall(nm) == nm
end

# issue #112
@test roundtripeq(Array, [1,2,3,4])
@test roundtripeq(Array{Int8}, [1,2,3,4])

# buffers

@pyimport array
@pyimport numpy as np

@test roundtripeq({1 2 3; 4 5 6})
@test roundtripeq([])
@test convert(Array{PyAny,1}, PyObject({1 2 3; 4 5 6})) == {{1,2,3},{4,5,6}}
@test roundtripeq(begin A = Array(Int, 3); A[1] = 1; A[2] = 2; A[3] = 3; A; end)
@test convert(PyAny, PyObject(begin A = Array(Any); A[1] = 3; A; end)) == 3

array2py2arrayeq(x) = PyCall.py2array(Float64, PyCall.array2py(x)) == x
@test array2py2arrayeq(rand(3))
@test array2py2arrayeq(rand(3,4))
@test array2py2arrayeq(rand(3,4,5))

let b = PyCall.PyBuffer(pyutf8("test string"))
    @test ndims(b) == 1
    @test (length(b),) == (length("test string"),) == (size(b, 1),) == size(b)
    @test stride(b, 1) == 1
    @test PyCall.iscontiguous(b) == true
end

# Check 1D arrays
a = array.array("f", [1,2,3])
view = PyCall.pygetbuffer(a, PyCall.PyBUF_RECORDS)
@test ndims(view) == 1
@test length(view) == 3
@test sizeof(view) == sizeof(Float32) * 3
@test size(view) == (3,)
@test strides(view) == (1,)
@test PyCall.aligned(view) == true
# a vector is both c/f contiguous
@test PyCall.c_contiguous(view) == true
@test PyCall.f_contiguous(view) == true
@test PyCall.pyfmt(view) == "f"

# Check 1D numpy arrays
a = np.array([1.0,2.0,3.0])
@test a[:dtype] == np.dtype("float64")
view = PyCall.pygetbuffer(a, PyCall.PyBUF_RECORDS)
@test ndims(view) == 1
@test length(view) == 3
@test sizeof(view) == a[:nbytes]
@test size(view) == (3,)
@test strides(view) == (1,)
@test PyCall.aligned(view) == true
# a vector is both c/f contiguous
@test PyCall.c_contiguous(view) == true
@test PyCall.f_contiguous(view) == true
@test PyCall.pyfmt(view) == "d"

# Check 2D C ordered arrays
a = np.array([[1 2 3],
	          [1 2 3]])
@test a[:dtype] == np.dtype("int64")
view = PyCall.pygetbuffer(a, PyCall.PyBUF_RECORDS)
@test ndims(view) == 2
@test length(view) == 6
@test sizeof(view) == a[:nbytes]
@test size(view) == (2,3)
@test strides(view) == (3,1)
@test PyCall.aligned(view) == true
@test PyCall.c_contiguous(view) == true
@test PyCall.f_contiguous(view) == false
@test PyCall.pyfmt(view) == "l"

# Check Multi-D C ordered arrays
a = np.ones((10,5,3), dtype="float32", order="C")
@test a[:dtype] == np.dtype("float32")
view = PyCall.pygetbuffer(a, PyCall.PyBUF_RECORDS)
@test ndims(view) == 3
@test length(view) == (10 * 5 * 3)
@test sizeof(view) == a[:nbytes]
@test size(view) == (10, 5, 3)
@test strides(view) == (5 * 3, 3, 1)
@test PyCall.aligned(view) == true
@test PyCall.c_contiguous(view) == true
@test PyCall.f_contiguous(view) == false
@test PyCall.pyfmt(view) == "f"

# Check Multi-D F ordered arrays
a = np.ones((10,5,3), dtype="uint8", order="F")
@test a[:dtype] == np.dtype("uint8")
view = PyCall.pygetbuffer(a, PyCall.PyBUF_RECORDS)
@test ndims(view) == 3
@test length(view) == (10 * 5 * 3)
@test sizeof(view) == a[:nbytes]
@test size(view) == (10, 5, 3)
@test strides(view) == (1, 10, 50)
@test PyCall.aligned(view) == true
@test PyCall.c_contiguous(view) == false
@test PyCall.f_contiguous(view) == true
@test PyCall.pyfmt(view) == "B"

###################################################
# PyArray Tests

npa = np.array([1,2,3])
a = PyArray(npa)

@test size(a) == (3,)
@test ndims(a) == 1
@test strides(a) == (1,)
@test size(similar(a)) == size(a)
@test eltype(similar(a)) == eltype(a)
@test summary(a) == "3-element Int64 PyArray"
@test sprint() do io; show(io, a); end == "[1,2,3]"

a[1] = 3
a[2] = 2
a[3] = 1
@test sprint() do io; show(io, npa); end == "PyObject array([3, 2, 1])"
aa = np.asarray(a)
@test np.shape(aa) == size(a)

npa = np.ones((10,10), dtype="float32")
a = PyArray(npa)

@test size(a) == (10,10)
@test ndims(a) == 2
@test strides(a) == (10, 1)
@test size(similar(a)) == size(a)
@test eltype(similar(a)) == eltype(a)
@test summary(a) == "10x10 Float32 PyArray"
@test sprint() do io; show(io, a); end == sprint() do io
				              show(io, ones(Float32, (10,10)))
				      	  end
aa = np.asarray(a)
@test np.shape(aa) == size(a)

npa = np.ones((10,10,20), dtype="float32")
a = PyArray(npa)
@test size(a) == (10,10,20)
@test strides(a) == (20 * 10, 20, 1)
@test ndims(a) == 3
@test size(similar(a)) == size(a)
@test eltype(similar(a)) == eltype(a)
@test summary(a) == "10x10x20 Float32 PyArray"

aa = np.asarray(a)
@test np.shape(aa) == size(a)

# Test fail because of show uses unimplemented subarray methods
#@test sprint() do io; show(io, a); end == sprint() do io
#				              show(io, ones(Float32, (10,10,20)))
#				      	  end

###################################################
# PyObject Conversion
npa = np.ones(10, dtype="float32")
@test PyObject(a) == a.o

a = convert(PyArray, npa)
@test typeof(a) === PyArray{Float32,1}

# PyArray conversion shares the underlying buffer
a[5] = 10
@test pycall(npa["__getitem__"], Float64, 4) == 10.0

npa = np.ones(10, dtype="float32")
a = convert(Array{Float32,1}, npa)
@test typeof(a) === Array{Float32,1}

# Array conversion should perform a copy
a[5] = 10
@test pycall(npa["__getitem__"], Float64, 4) == 1.0

#TODO: Multi-D conversion
#npa = np.ones((10,10), dtype="float32")
#aa = convert(Array{Float32,2}, npa)
#@test typeof(aa) === Array{Float32,2}
#aa[5,5] = 10
#@show pycall(npa["__getitem__"], PyCall.PyAny, (4,4))

###################################################
#TODO: record array support

a = np.zeros((2,), dtype=("i4,f4,a10"))
view = PyCall.pygetbuffer(a, PyCall.PyBUF_FULL)
@test PyCall.pyfmt(view) == "T{=i:f0:f:f1:10s:f2:}"

a = np.zeros(3, dtype="3int8, float32, (2,3)float64")
view = PyCall.pygetbuffer(a, PyCall.PyBUF_FULL)
@test PyCall.pyfmt(view) == "T{(3)b:f0:=f:f1:(2,3)d:f2:}"

a = np.zeros(3, dtype={"names" => ["col1", "col2"],
                       "formats" => ["i4", "f4"]})
view = PyCall.pygetbuffer(a, PyCall.PyBUF_FULL)
@test PyCall.pyfmt(view) == "T{i:col1:f:col2:}"

immutable Test1
   a::Float32
   b::Float64
   c::Bool
end

@test isbits(Test1)
@test PyCall.jltype_to_pyfmt(Test1) == "T{f:a:d:b:?:c:}"

immutable Test2
  a::Float32
  b::Test1
end

@test isbits(Test2)
@test PyCall.jltype_to_pyfmt(Test2) == "T{f:a:T{f:a:d:b:?:c:}}"

type Test3{T}
   a::T
   b::T
end
@test PyCall.jltype_to_pyfmt(Test3{Float32}) == "T{f:a:f:b:}"

type Dummy end
type Test4
   a::Dummy
end
@test_throws PyCall.jltype_to_pyfmt(Test4)

immutable Test5
end
@test_throws PyCall.jltype_to_pyfmt(Test5)
