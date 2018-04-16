using Compat.Test, PyCall, Compat
using PyCall: f_contiguous, PyBUF_ND_CONTIGUOUS, array_format

pyutf8(s::PyObject) = pycall(s["encode"], PyObject, "utf-8")
pyutf8(s::String) = pyutf8(PyObject(s))

@testset "PyBuffer" begin
    np = pyimport("numpy")
    nparr = np["array"]
    listpy = pybuiltin("list")
    arrpyo(args...; kwargs...)  = pycall(nparr, PyObject, args...; kwargs...)
    listpyo(args...) = pycall(listpy, PyObject, args...)
    pytestarray(sz::Int...; order="C") =
        pycall(arrpyo(1.0:prod(sz), "d")["reshape"], PyObject, sz, order=order)

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
        @test_throws ArgumentError NoCopyArray(wrong_endian_arr)
        @test_throws ArgumentError PyArray(wrong_endian_arr)
    end

    @testset "NoCopyArray 1d" begin
        ao = arrpyo(1.0:10.0, "d")
        pybuf = PyBuffer(ao, PyBUF_ND_CONTIGUOUS)
        T, native_byteorder = array_format(pybuf)
        @test T == Float64
        @test native_byteorder == true
        @test size(pybuf) == (10,)
        @test strides(pybuf) == (1,) .* sizeof(T)
        nca = NoCopyArray(ao)
        @test !(nca isa PermutedDimsArray)
        @test nca isa Array
        @test nca[3] == ao[3]
        @test nca[4] == ao[4]
    end

    @testset "NoCopyArray 2d f-contig" begin
        ao = arrpyo(reshape(1.0:12.0, (3,4)), "d", order="F")
        pybuf = PyBuffer(ao, PyBUF_ND_CONTIGUOUS)
        T, native_byteorder = array_format(pybuf)
        @test T == Float64
        @test native_byteorder == true
        @test size(pybuf) == (3,4)
        @test strides(pybuf) == (1, 3) .* sizeof(T)
        nca = NoCopyArray(ao)
        @test !(nca isa PermutedDimsArray)
        # @show typeof(nca) (nca isa Array)
        @test nca isa Array
        @test size(nca) == (3,4)
        @test strides(nca) == (1,3)
        @test nca[3,2] == ao[3,2]
        @test nca[2,3] == ao[2,3]
    end

    @testset "NoCopyArray 3d c-contig" begin
        ao = pytestarray(3,4,5)
        pybuf = PyBuffer(ao, PyBUF_ND_CONTIGUOUS)
        T, native_byteorder = array_format(pybuf)
        @test T == Float64
        @test native_byteorder == true
        @test size(pybuf) == (3,4,5)
        @test strides(pybuf) == (20,5,1) .* sizeof(T)
        nca = NoCopyArray(ao)
        @test nca isa PermutedDimsArray
        @test !(nca isa Array)
        @test size(nca) == (3,4,5)
        @test strides(nca) == (20,5,1)
        @test nca[2,3,4] == ao[2,3,4]
        @test nca[3,2,4] == ao[3,2,4]
    end

    @testset "isbuftype" begin
        @test isbuftype(PyObject(0)) == false
        @test isbuftype(listpyo((1.0:10.0...))) == false
        @test isbuftype(arrpyo(1.0:10.0, "d")) == true
        @test isbuftype(PyObject([1:10...])) == true
    end
end
