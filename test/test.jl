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

@test roundtripeq({1 2 3; 4 5 6})
@test roundtripeq([])
@test convert(Array{PyAny,1}, PyObject({1 2 3; 4 5 6})) == {{1,2,3},{4,5,6}}
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
