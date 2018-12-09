using Compat.Test, PyCall, Compat
using PyCall: f_contiguous, PyBUF_ND_CONTIGUOUS, array_format, npy_initialized,
NoCopyArray, isbuftype, setdata!

pyutf8(s::PyObject) = pycall(s["encode"], PyObject, "utf-8")
pyutf8(s::String) = pyutf8(PyObject(s))

@testset "PyBuffer" begin
    @testset "String Buffers" begin
        b = PyCall.PyBuffer(pyutf8("test string"))
        @test ndims(b) == 1
        @test (length(b),) == (length("test string"),) == (size(b, 1),) == size(b)
        @test stride(b, 1) == 1
        @test PyCall.iscontiguous(b) == true
    end

    if !npy_initialized
        println("Skipping array related buffer tests since NumPy not available")
    else
        np = pyimport("numpy")
        listpy = pybuiltin("list")
        arrpyo(args...; kwargs...) =
            pycall(np["array"], PyObject, args...; kwargs...)
        listpyo(args...) = pycall(listpy, PyObject, args...)
        pytestarray(sz::Int...; order="C") =
            pycall(arrpyo(1.0:prod(sz), "d")["reshape"], PyObject, sz, order=order)

        @testset "Non-native-endian" begin
            wrong_endian_str = ENDIAN_BOM == 0x01020304 ? "<" : ">"
            wrong_endian_arr =
                pycall(np["ndarray"], PyObject, 2; buffer=UInt8[0,1,3,2],
                                                   dtype=wrong_endian_str*"i2")
            # Not supported, so throws
            @test_throws ArgumentError NoCopyArray(wrong_endian_arr)
            @test_throws ArgumentError PyArray(wrong_endian_arr)
        end

        @testset "dtype should match eltype" begin
            npy2jl = Dict(np["int64"][:__name__]=>Int64,
                          np["int32"][:__name__]=>Int32)
            nparr = arrpyo(1:10)
            jltype = npy2jl[pystr(nparr["dtype"])]
            @test eltype(convert(PyAny, nparr)) == jltype
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
            ao = arrpyo(reshape(1.0:12.0, (3,4)) |> collect, "d", order="F")
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
            @test isbuftype(listpyo((1.0:10.0...,))) == false
            @test isbuftype(arrpyo(1.0:10.0, "d")) == true
            @test isbuftype(PyObject([1:10...])) == true
        end

        # TODO maybe move these to a test_pyarray.jl
        @testset "setdata!" begin
            ao1 = arrpyo(1.0:10.0, "d")
            pyarr = convert(PyArray, ao1)
            ao2 = arrpyo(11.0:20.0, "d")
            setdata!(pyarr, ao2)
            @test all(pyarr[1:10] .== 11.0:20.0)
        end

        @testset "similar on PyArray PyVec getindex" begin
            jlarr1 = [1:10;]
            jlarr2 = hcat([1:10;], [1:10;])
            pyarr1 = pycall(np["array"], PyArray, jlarr1)
            pyarr2 = pycall(np["array"], PyArray, jlarr2)
            @test all(pyarr1[1:10]    .== jlarr1[1:10])
            @test all(pyarr2[1:10, 2] .== jlarr2[1:10, 2])
            @test all(pyarr2[1:10, 1:2] .== jlarr2)
        end
    end
end
