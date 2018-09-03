using PyCall, Compat
using Compat.Test, Compat.Dates, Compat.Serialization

filter(f, itr) = collect(Iterators.filter(f, itr))
filter(f, d::AbstractDict) = Base.filter(f, d)

PYTHONPATH=get(ENV,"PYTHONPATH","")
PYTHONHOME=get(ENV,"PYTHONHOME","")
PYTHONEXECUTABLE=get(ENV,"PYTHONEXECUTABLE","")
Compat.@info "Python version $pyversion from $(PyCall.libpython), PYTHONHOME=$(PyCall.PYTHONHOME)\nENV[PYTHONPATH]=$PYTHONPATH\nENV[PYTHONHOME]=$PYTHONHOME\nENV[PYTHONEXECUTABLE]=$PYTHONEXECUTABLE"

roundtrip(T, x) = convert(T, PyObject(x))
roundtrip(x) = roundtrip(PyAny, x)
roundtripeq(T, x) = roundtrip(T, x) == x
roundtripeq(x) = roundtrip(x) == x

@pyimport math

struct TestConstruct
    x
end

pymodule_exists(s::AbstractString) = !ispynull(pyimport_e(s))

# default integer type for PyAny conversions
const PyInt = pyversion < v"3" ? Int : Clonglong

@testset "PyCall" begin
    # conversion of NumPy scalars before npy_initialized by array conversions (#481)
    np = pyimport_e("numpy")
    if !ispynull(np) # numpy is installed, so test
        let o = get(pycall(np["array"], PyObject, 1:3), PyObject, 2)
            @test convert(Int32, o) === Int32(3)
            @test convert(Int64, o) === Int64(3)
            @test convert(Float64, o) === Float64(3)
            @test convert(Complex{Int}, o) === 3+0im
        end
    end

    # test handling of type-tuple changes in Julia 0.4
    import PyCall.pyany_toany
    @test pyany_toany(Int) == Int
    @test pyany_toany(PyAny) == Any
    @test pyany_toany(Tuple{Int,PyAny}) == Tuple{Int,Any}
    @test pyany_toany(Tuple{Int,Tuple{PyAny,Int8}}) == Tuple{Int,Tuple{Any,Int8}}
    @test pyany_toany(Tuple{PyAny,Int,Vararg{PyAny}}) == Tuple{Any,Int,Vararg{Any}}

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
    @test roundtripeq(C_NULL) && roundtripeq(convert(Ptr{Cvoid}, 12345))
    @test roundtripeq([1,3,4,5]) && roundtripeq([1,3.2,"hello",true])
    @test roundtripeq([1 2 3;4 5 6]) && roundtripeq([1. 2 3;4 5 6])
    @test roundtripeq((1,(3.2,"hello"),true)) && roundtripeq(())
    @test roundtripeq(Int32)
    @test roundtripeq(Dict(1 => "hello", 2 => "goodbye")) && roundtripeq(Dict())
    @test roundtripeq(UInt8[1,3,4,5])
    @test roundtrip(3 => 4) == (3,4)
    @test roundtrip(Pair{Int,Int}, 3 => 4) == Pair(3,4)
    @test eltype(roundtrip([Ref(1), Ref(2)])) == typeof(Ref(1))

    @test pycall(PyObject(x -> x + 1), PyAny, 314158) == 314159
    @test PyObject(x -> x + 1)(314158) == 314159
    @test PyAny(PyObject(3)) == 3
    @test roundtrip(x -> x + 1)(314158) == 314159

    testkw(x; y=0) = x + 2*y
    @test pycall(PyObject(testkw), PyAny, 314157) == 314157
    @test pycall(PyObject(testkw), PyAny, 314157, y=1) == 314159
    @test roundtrip(testkw)(314157) == 314157
    @test roundtrip(testkw)(314157, y=1) == 314159

    # check type stability of pycall with an explicit return type
    @inferred pycall(PyObject(1)[:__add__], Int, 2)

    if PyCall.npy_initialized
        @test PyArray(PyObject([1. 2 3;4 5 6])) == [1. 2 3;4 5 6]
        let A = rand(Int, 2,3,4), B = rand(Bool, 2,3,4)
            @test convert(PyAny, PyReverseDims(A)) == permutedims(A, [3,2,1])
            @test convert(PyAny, PyReverseDims(BitArray(B))) == permutedims(B, [3,2,1])
        end
    end
    @test PyVector(PyObject([1,3.2,"hello",true])) == [1,3.2,"hello",true]
    @test PyDict(PyObject(Dict(1 => "hello", 2 => "goodbye"))) == Dict(1 => "hello", 2 => "goodbye")
    @test roundtripeq(BitArray([true, false, true, true]))

    let d = PyDict(Dict(1 => "hello", 34 => "yes" ))
        @test get(d.o, 1) == "hello"
        set!(d.o, 34, "goodbye")
        @test d[34] == "goodbye"
        @test sort!(keys(Int, d)) == sort!(collect(d.o[:keys]())) == sort!(collect(keys(d))) == [1, 34]
        @test eltype(d) == eltype(typeof(d)) == Pair{Int, String}
    end

    let d = Dict(zip(1:1000, 1:1000)), f
        f(k,v) = iseven(k) # For 0.6
        f(kv) = iseven(kv[1]) # For 0.7
        @test filter(f, d) == filter(f, PyDict(d)) == filter!(f, PyDict(d)) ==
              Dict(zip(2:2:1000, 2:2:1000))
    end

    @test roundtripeq(Any[1 2 3; 4 5 6])
    @test roundtripeq([])
    @test convert(Array{PyAny,1}, PyObject(Any[1 2 3; 4 5 6])) == Any[Any[1,2,3],Any[4,5,6]]
    if PyCall.npy_initialized
        @test roundtripeq(begin A = Array{Int}(undef); A[1] = 3; A; end)
    end
    @test convert(PyAny, PyObject(begin A = Array{Any}(undef); A[1] = 3; A; end)) == 3

    array2py2arrayeq(x) = PyCall.py2array(Float64,PyCall.array2py(x)) == x
    @test array2py2arrayeq(rand(3))
    @test array2py2arrayeq(rand(3,4))
    @test array2py2arrayeq(rand(3,4,5))

    @test roundtripeq(2:10) && roundtripeq(10:-1:2)
    @test roundtrip(2:2.0:10) == convert(Vector{Float64}, 2:2.0:10)

    @test math.sin(3) ≈ sin(3)

    @test collect(PyObject([1,"hello",5])) == [1,"hello",5]

    @test try @eval (@pyimport os.path) catch ex
        if VERSION >= v"0.7.0-DEV.1729"
            ex = (ex::LoadError).error
        end
        isa(ex, ArgumentError)
    end

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
        py"lambda x: x.append(\"bar\")"(x)
        @test x == ["bar"]
    end

    @test roundtripeq(Dates.Date(2012,3,4))
    @test roundtripeq(Dates.DateTime(2012,3,4, 7,8,9,11))
    @test roundtripeq(Dates.Millisecond(typemax(Int32)))
    @test roundtripeq(Dates.Millisecond(typemin(Int32)))
    @test roundtripeq(Dates.Second, Dates.Second(typemax(Int32)))
    @test roundtripeq(Dates.Second, Dates.Second(typemin(Int32)))
    @test roundtripeq(Dates.Day, Dates.Day(999999999)) # max allowed day timedelta
    @test roundtripeq(Dates.Day, Dates.Day(-999999999)) # min allowed day timedelta

    # fixme: is there any nontrivial showable test we can do?
    @test !showable("text/html", PyObject(1))

    # in Python 3, we need a specific encoding to write strings or bufferize them
    # (http://stackoverflow.com/questions/5471158/typeerror-str-does-not-support-the-buffer-interface)
    pyutf8(s::PyObject) = pycall(s["encode"], PyObject, "utf-8")
    pyutf8(s::String) = pyutf8(PyObject(s))

    # IO (issue #107)
    #@test roundtripeq(stdout) # No longer true since #250
    let buf = Compat.IOBuffer(read=false, write=true), obuf = PyObject(buf)
        @test !obuf[:isatty]()
        @test obuf[:writable]()
        @test !obuf[:readable]()
        @test obuf[:seekable]()
        obuf[:write](pyutf8("hello"))
        obuf[:flush]()  # should be a no-op, since there's no flushing IOBuffer
        @test position(buf) == obuf[:tell]() == 5
        let p = obuf[:seek](-2, 1)
            @test p == position(buf) == 3
        end
        let p = obuf[:seek](0, 0)
            @test p == position(buf) == 0
        end
        @test String(take!(buf)) == "hello"
        obuf[:writelines](["first\n", "second\n", "third"])
        @test String(take!(buf)) == "first\nsecond\nthird"
        obuf[:write](b"möre stuff")
        @test String(take!(buf)) == "möre stuff"
        @test isopen(buf) == !obuf[:closed] == true
        obuf[:close]()
        @test isopen(buf) == !obuf[:closed] == false
    end
    let buf = IOBuffer("hello\nagain"), obuf = PyObject(buf)
        @test !obuf[:writable]()
        @test obuf[:readable]()
        @test obuf[:readlines]() == ["hello\n", "again"]
    end
    let buf = IOBuffer("hello\nagain"), obuf = PyObject(buf)
        @test codeunits(obuf[:read](5)) == b"hello"
        @test codeunits(obuf[:readall]()) == b"\nagain"
    end
    let buf = IOBuffer("hello\nagain"), obuf = PyTextIO(buf)
        @test obuf[:encoding] == "UTF-8"
        @test obuf[:read](3) == "hel"
        @test obuf[:readall]() == "lo\nagain"
    end
    let nm = tempname()
        open(nm, "w") do f
            # @test roundtripeq(f)  # PR #250
            pf = PyObject(f)
            @test pf[:fileno]() == fd(f)
            @test pf[:writable]()
            @test !pf[:readable]()
            pf[:write](pyutf8(nm))
            pf[:flush]()
        end
        @test read(nm, String) == nm
    end

    # issue #112
    @test roundtripeq(Array, [1,2,3,4])
    @test roundtripeq(Array{Int8}, [1,2,3,4])

    # conversion of numpy scalars
    pyanycheck(x::Any) = pyanycheck(typeof(x), PyObject(x))
    pyanycheck(T, o::PyObject) = isa(convert(PyAny, o), T)
    @test pyanycheck(PyInt, PyVector{PyObject}(PyObject([1]))[1])
    @test pyanycheck(Float64, PyVector{PyObject}(PyObject([1.3]))[1])
    @test pyanycheck(ComplexF64, PyVector{PyObject}(PyObject([1.3+1im]))[1])
    @test pyanycheck(Bool, PyVector{PyObject}(PyObject([true]))[1])

    # conversions of Int128 and BigInt
    let i = 1234567890123456789 # Int64
        @test PyObject(i) - i == 0
    end
    let i = 12345678901234567890 # Int128
        @test PyObject(i) - i == 0
    end
    let i = BigInt(12345678901234567890), o = PyObject(i) # BigInt
        @test o - i == 0
        @test convert(BigInt, o) == i
        if pyversion >= v"3.2"
            @test PyAny(o) == i == convert(Integer, o)
            @test_throws InexactError convert(Int64, o)
        end
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
        @test popfirst!(o) == 8 && collect(o) == [3]
        @test collect(pushfirst!(o, 9)) == [9,3]
        @test collect(prepend!(o, [5,4,2])) == [5,4,2,9,3]
        @test collect(append!(o, [1,6,8])) == [5,4,2,9,3,1,6,8]
        @test isempty(empty!(o))
    end
    let o = PyObject(Any[8,3])
        @test collect(append!(o, o)) == [8,3,8,3]
        push!(o, 1)
        @test collect(prepend!(o, o)) == [8,3,8,3,1,8,3,8,3,1]
    end

    # issue #216:
    @test length(collect(pyimport("itertools")[:combinations]([1,2,3],2))) == 3

    # PyNULL and copy!
    let x = PyNULL(), y = copy!(x, PyObject(314159))
        @test convert(Int, x) == convert(Int, y) == 314159
    end
    @test ispynull(PyNULL())
    @test !ispynull(PyObject(3))
    @test ispynull(pydecref(PyObject(3)))

    @test !ispynull(pyimport_conda("inspect", "not a conda package"))
    import Conda
    if PyCall.conda
        # import pyzmq to test PR #294
        let already_installed = "pyzmq" ∈ Conda._installed_packages()
            @test !ispynull(pyimport_conda("zmq", "pyzmq"))
            @test "pyzmq" ∈ Conda._installed_packages()
            if !already_installed
                Conda.rm("pyzmq")
            end
        end
    end

    let x = 7
        py"""
        def myfun(x):
            return x + $x
        """
        @test py"1 + 2" == 3
        @test py"1 + $x" == 8
        @test py"1 + $(x^2)" == 50
        @test py"myfun"(10) == 17
    end

    # issue #352
    let x = "1+1"
        @test py"$x" == "1+1"
        @test py"$$x" == py"$$(x)" == 2
        @test py"7 - $$x - 7" == 0 # evaluates "7 - 1 + 1 - 7"
        @test py"7 - ($$x) - 7" == -2 # evaluates "7 - (1 + 1) - 7"
        @test py"1 + $$(x[1:2]) 3" == 5 # evals 1 + 1+ 3
    end

    # Float16 support:
    if PyCall.npy_initialized
        @test roundtripeq(Float16[17 18 Inf -Inf -0.0 0.0])
        @test isa(roundtrip(Float16[17]), Vector{Float16})
    end

    """
    foobar doc
    """
    foobar(x) = x+1

    # function attributes
    let o = PyObject(foobar)
        @test o[:__name__] == o[:func_name] == string(foobar)
        @test o[:__doc__] == o[:func_doc] == "foobar doc\n"
        @test o[:__module__] == o[:__defaults__] == o[:func_defaults] ==
              o[:__closure__] == o[:func_closure] == nothing
    end

    # issue #345
    let weakdict = pyimport("weakref")["WeakValueDictionary"]
        # (use weakdict for the value, since Python supports
        #  weak references to type objects)
        @test convert(Dict{Int,PyObject}, weakdict(Dict(3=>weakdict))) == Dict(3=>weakdict)
        @test get(weakdict(Dict(3=>weakdict)),3) == weakdict
    end

    # Expose python docs to Julia doc system
    py"""
    def foo():
        "foo docstring"
        return 0
    """
    global foo354 = py"foo"
    # use 'content' since `Text` objects test equality by object identity
    @test @doc(foo354).content == "foo docstring"

    # binary operators
    for b in (4, PyObject(4))
        for op in (+, -, *, /, %, &, |, ^, <<, >>, ⊻)
            let x = op(PyObject(111), b)
                @test isa(x, PyObject)
                @test convert(PyAny, x) == op(111, 4)
            end
            @test convert(PyAny, op(b, PyObject(3))) == op(4, 3)
        end
    end
    @test convert(PyAny, PyObject(3)^4)  == 3^4 # literal integer powers
    @test convert(PyAny, PyObject(3)^0)  == 1   # literal integer powers
    @test convert(PyAny, PyObject(2)^-1) == 0.5 # literal integer powers
    # unary operators
    for op in (+, -, ~, abs)
        let x = op(PyObject(-3))
            @test isa(x, PyObject)
            @test convert(PyAny, x) == op(-3)
        end
    end
    # comparisons
    for x in (3,4,5), y in (3.0,4.0,5.0)
        for op in (<, <=, ==, !=, >, >=, isless, isequal)
            @test op(PyObject(x), PyObject(y)) == op(x, y)
            if op != isequal
                @test op(PyObject(x), y) == op(x, y)
            end
        end
    end

    # updating operators .+= etcetera
    let o = PyObject(Any[1,2]), c = o
        broadcast!(+, o, o, Any[3,4]) # o .+= x doesn't work yet in 0.7
        @test collect(o) == [1,2,3,4]
        @test o.o == c.o # updated in-place
    end

    # more flexible bool conversions, matching Python "truth value testing"
    @test convert(Bool, PyObject(nothing)) === false
    @test convert(Bool, PyObject(0.0)) === false
    @test convert(Bool, PyObject(Any[])) === false
    @test convert(Bool, PyObject(17.3)) === true
    @test convert(Bool, PyObject(Any[0])) === true
    @test convert(Bool, PyVector{PyObject}(PyObject([false]))[1]) === false

    # serialization
    let py_sum_obj = pybuiltin("sum")
        b = IOBuffer()
        serialize(b, py_sum_obj)
        @test py_sum_obj == deserialize(seekstart(b))

        b = IOBuffer()
        serialize(b, PyNULL())
        @test PyNULL() == deserialize(seekstart(b))
    end

    # @pycall macro expands correctly
    _pycall = GlobalRef(PyCall,:pycall)
    @test macroexpand(@__MODULE__, :(@pycall foo(bar)::T)) == :($(_pycall)(foo, T, bar))
    @test macroexpand(@__MODULE__, :(@pycall foo(bar, args...)::T)) == :($(_pycall)(foo, T, bar, args...))
    @test macroexpand(@__MODULE__, :(@pycall foo(bar; kwargs...)::T)) == :($(_pycall)(foo, T, bar; kwargs...))

    # basic @pywith functionality
    fname = tempname()
    try
        @test begin
            @pywith pybuiltin("open")(fname,"w") as f begin
                f[:write]("test")
            end
            open(io->read(io, String), fname)=="test"
        end
    finally
        rm(fname,force=true)
    end

    @test occursin("integer", Base.Docs.doc(PyObject(1)).content)
    @test occursin("no docstring", Base.Docs.doc(PyObject(py"lambda x: x+1")).content)

    let b = rand(UInt8, 1000)
        @test(convert(Vector{UInt8}, pybytes(b)) == b
              == convert(Vector{UInt8}, pybytes(String(copy(b))))
              == convert(Vector{UInt8}, pybytes(codeunits(String(copy(b))))))
    end

    let t = convert(Tuple, PyObject((3,34)))
        @test isa(t, Tuple{PyObject,PyObject})
        @test t == (PyObject(3), PyObject(34))
    end
    for T in (Tuple{Vararg{PyAny}}, NTuple{2,PyInt}, Tuple{PyInt,PyInt}, Tuple{Vararg{PyInt}}, Tuple{PyInt,Vararg{PyInt}})
        let t = convert(T, PyObject((3,34)))
            @test isa(t, Tuple{PyInt,PyInt})
            @test t == (3,34)
        end
    end
    @test_throws BoundsError convert(NTuple{3,Int}, PyObject((3,34)))

    let p = PyCall.pickle(), buf = IOBuffer()
        p[:dump]("hello world", buf)
        p[:dump](314159, buf)
        p[:dump](Any[1,1,2,3,5,8], buf)
        @test p[:load](seekstart(buf)) == "hello world"
        @test p[:load](buf) == 314159
        @test p[:load](buf) == [1,1,2,3,5,8]
    end

    # Test that we can call constructors on the python side
    @test pycall(PyObject(TestConstruct), PyAny, 1).x == 1

    # Test getattr fallback
    @test PyObject(TestConstruct(1))[:x] == 1
    @test_throws KeyError PyObject(TestConstruct(1))[:y]

    # iterating over Julia objects in Python:
    @test py"[x**2 for x in $(PyCall.pyjlwrap_new(1:4))]" ==
          py"[x**2 for x in $(x for x in 1:4)]" ==
          py"[x**2 for x in $(PyCall.jlwrap_iterator(1:4))]" ==
          [1,4,9,16]

    let o = PyObject("foo")
        @test pystr(o) == "foo"
        @test pyrepr(o) == "'foo'"
    end

    # pyfunction
    @test pyfunction(factorial, Int)(3) === PyInt(6)
    @test pyfunction(sin, Complex{Int})(3) === sin(3+0im)
    @test pyfunctionret(factorial, Float64, Int)(3) === 6.0
    @test pyfunctionret(factorial, nothing, Int)(3) === nothing
    @test PyCall.is_pyjlwrap(pycall(pyfunctionret(factorial, Any, Int), PyObject, 3))
    @test pyfunctionret(max, Int, Vararg{Int})(3,4,5) === PyInt(5)

    # broadcasting scalars
    let o = PyObject(3) .+ [1,4]
        @test o isa Vector{PyObject}
        @test o == [4,7]
    end

    # issue #533
    @test py"lambda x,y,z: (x,y,z)"(3:6,4:10,5:11) === (PyInt(3):PyInt(6), PyInt(4):PyInt(10), PyInt(5):PyInt(11))
end

######################################################################
#@pydef tests: type declarations need to happen at top level

# issue #389
@pydef mutable struct EmptyClass
end

# @pywith errors correctly handled
@pydef mutable struct IgnoreError
    function __init__(self, ignore)
        self[:ignore] = ignore
    end
    __enter__(self) = ()
    __exit__(self, typ, value, tb) = self[:ignore]
end

# @pydef example from README
@pydef mutable struct Doubler <: PyCall.builtin[:AssertionError]
    __init__(self, x=10) = (self[:x] = x)
    function my_method(self, arg1::Number)
        return arg1 + 20
    end
    type_str(self, obj::T) where T = string(T)
    x2.get(self) = self[:x] * 2
    function x2.set!(self, new_val)
        self[:x] = new_val / 2
    end
end

@testset "pydef" begin
    d = Doubler(5)
    @test d[:x] == 5
    d[:x2] = 30
    @test d[:x] == 15
    @test d[:type_str](10) == string(PyInt)
    @test PyCall.builtin[:isinstance](d, PyCall.builtin[:AssertionError])

    @test_throws ErrorException @pywith IgnoreError(false) error()
    @test (@pywith IgnoreError(true) error(); true)
end

@testset "callback" begin
    # Returning existing PyObject in Julia should not invalidate it.
    # https://github.com/JuliaPy/PyCall.jl/pull/552
    anonymous = Module()
    Base.eval(
        anonymous, quote
            using PyCall
            obj = pyimport("sys")  # get some PyObject
        end)
    py"""
    ns = {}
    def set(name):
        ns[name] = $include_string($anonymous, name)
    """
    py"set"("obj")
    @test anonymous.obj != PyNULL()

    # Test above for pyjlwrap_getattr too:
    anonymous = Module()
    Base.eval(
        anonymous, quote
            using PyCall
            struct S
                x
            end
            obj = S(pyimport("sys"))
        end)
    py"""
    ns = {}
    def set(name):
        ns[name] = $include_string($anonymous, name).x
    """
    py"set"("obj")
    @test anonymous.obj.x != PyNULL()

    # Test above for pyjlwrap_iternext too:
    anonymous = Module()
    Base.eval(
        anonymous, quote
            using PyCall
            sys = pyimport("sys")
            obj = (sys for _ in 1:1)
        end)
    py"""
    ns = {}
    def set(name):
        ns[name] = list(iter($include_string($anonymous, name)))
    """
    py"set"("obj")
    @test anonymous.sys != PyNULL()
end

include("test_pyfncall.jl")
