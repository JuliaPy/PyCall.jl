using Compat.Test, PyCall, Compat
using PyCall: f_contiguous, PyBUF_ND_CONTIGUOUS, array_format

pyutf8(s::PyObject) = pycall(s["encode"], PyObject, "utf-8")
pyutf8(s::String) = pyutf8(PyObject(s))

@testset "PyBuffer" begin
    np = pyimport("numpy")
    # arrpyo = pyimport("array")["array"]=>PyObject
    # listpyo = pybuiltin("list")=>PyObject
    arrpy = pyimport("array")["array"]
    listpy = pybuiltin("list")
    arrpyo(args...)  = pycall(arrpy, PyObject, args...)
    listpyo(args...) = pycall(listpy, PyObject, args...)

    @testset "String Buffers" begin
        b = PyCall.PyBuffer(pyutf8("test string"))
        @test ndims(b) == 1
        @test (length(b),) == (length("test string"),) == (size(b, 1),) == size(b)
        @test stride(b, 1) == 1
        @test PyCall.iscontiguous(b) == true
    end

    @testset "Non-native-endian" begin
        wrong_endian_str = ENDIAN_BOM == 0x01020304 ? "<" : ">"
        wrong_endian_arr =
            pycall(np["ndarray"], PyObject, 2; buffer=UInt8[0,1,3,2],
                                               dtype=wrong_endian_str*"i2")
        # Not supported, so throws
        @test_throws ArgumentError ArrayFromBuffer(wrong_endian_arr)
        @test_throws ArgumentError PyArrayFromBuffer(wrong_endian_arr)
    end

    @testset "ArrayFromBuffer" begin
        ao = arrpyo("d", 1.0:10.0)
        pybuf = PyBuffer(ao, PyBUF_ND_CONTIGUOUS)
        T, native_byteorder = array_format(pybuf)
        @test T == Float64
        @test native_byteorder == true
        @test size(pybuf) == (10,)
        @test strides(pybuf) == (sizeof(T),)
        @test !(ArrayFromBuffer(ao) isa PermutedDimsArray)
        @test ArrayFromBuffer(ao) isa Array
    end

    @testset "isbuftype" begin
        @test isbuftype(PyObject(0)) == false
        @test isbuftype(listpyo((1.0:10.0...))) == false
        @test isbuftype(arrpyo("d", 1.0:10.0)) == true
        @test isbuftype(PyObject([1:10...])) == true
    end
end
