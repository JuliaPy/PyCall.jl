using Test, PyCall
using PyCall: f_contiguous, PyBUF_ND_CONTIGUOUS, array_format, npy_initialized,
NoCopyArray, isbuftype, setdata!

pyutf8(s::PyObject) = pycall(s."encode", PyObject, "utf-8")
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
        println(stderr, "Warning: skipping array related buffer tests since NumPy not available")
    else
        np = pyimport("numpy")
        listpy = pybuiltin("list")
        arrpyo(args...; kwargs...) =
            pycall(np."array", PyObject, args...; kwargs...)
        listpyo(args...) = pycall(listpy, PyObject, args...)
        pytestarray(sz::Int...; order="C") =
            pycall(arrpyo(1.0:prod(sz), "d")."reshape", PyObject, sz, order=order)

        @testset "Non-native-endian" begin
            wrong_endian_str = ENDIAN_BOM == 0x01020304 ? "<" : ">"
            wrong_endian_arr =
                pycall(np."ndarray", PyObject, 2; buffer=UInt8[0,1,3,2],
                                                   dtype=wrong_endian_str*"i2")
            # Not supported, so throws
            @test_throws ArgumentError NoCopyArray(wrong_endian_arr)
            @test_throws ArgumentError PyArray(wrong_endian_arr)
        end

        @testset "dtype should match eltype" begin
            npy2jl = Dict("int32"=>Int32, "float32"=>Float32,
                          "int64"=>Int64, "float64"=>Float64)
            for (pytype, jltype) in npy2jl
                @testset "dtype $pytype should match eltype $jltype" begin
                    jlarr = jltype[1:10;]
                    nparr = arrpyo(jlarr, dtype=pytype)
                    @test pystr(nparr."dtype") == pytype
                    jlarr2 = convert(PyAny, nparr)
                    @test eltype(jlarr2) == jltype
                    @test jlarr2 == jlarr
                end
            end
        end

        # f_contiguous(T, sz, st)
        @testset "f_contiguous 1D" begin
            # contiguous case: stride == sizeof(T)
            @test f_contiguous(Float64, (4,), (8,)) == true
            # non-contiguous case: stride != sizeof(T)
            @test f_contiguous(Float64, (4,), (16,)) == false
        end

        @testset "f_contiguous 2D" begin
            # contiguous: st[1] == sizeof(T), st[2] == st[1]*sz[1]
            @test f_contiguous(Float64, (4, 2), (8, 32)) == true
            # non-contiguous: stride != sizeof(T), but st[2] == st[1]*sz[1]
            @test f_contiguous(Float64, (4, 2), (16, 64)) == false
            # non-contiguous: stride == sizeof(T), but st[2] != st[1]*sz[1]
            @test f_contiguous(Float64, (4, 2), (8, 64)) == false
        end

        @testset "copy f_contig 1d" begin
            apyo = arrpyo(1.0:10.0, "d")
            pyarr = PyArray(apyo)
            jlcopy = copy(pyarr)
            @test pyarr.f_contig == true
            @test pyarr.c_contig == true
            @test all(jlcopy .== pyarr)
            # check it's not aliasing the same data
            jlcopy[1] = -1.0
            @test pyarr[1] == 1.0
        end

        @testset "copy c_contig 2d" begin
            apyo = pytestarray(2,3) # arrpyo([[1,2,3],[4,5,6]], "d")
            pyarr = PyArray(apyo)
            jlcopy = copy(pyarr)
            @test pyarr.c_contig == true
            @test pyarr.f_contig == false
            # check all is in order
            for i in 1:size(pyarr, 1)
                for j in 1:size(pyarr, 1)
                    @test jlcopy[i,j] == pyarr[i,j] == pyarr[i,j,1] == pyarr[CartesianIndex(i,j)]
                end
            end
            # check it's not aliasing the same data
            jlcopy[1,1] = -1.0
            @test pyarr[1,1] == 1.0
        end

        @testset "Non contiguous PyArrays" begin
            @testset "1d non-contiguous" begin
                # create an array of four Int32s, with stride 8
                nparr = pycall(np."ndarray", PyObject, 4,
                                buffer=UInt32[1,0,1,0,1,0,1,0],
                                dtype="i4", strides=(8,))
                pyarr = PyArray(nparr)

                # The convert goes via a PyArray then a `copy`
                @test convert(PyAny, nparr) == [1, 1, 1, 1]

                @test eltype(pyarr) == Int32
                @test sizeof(eltype(pyarr)) == 4
                @test pyarr.info.st == (8,)
                # not f_contig because not contiguous
                @test pyarr.f_contig == false
                @test copy(pyarr) == Int32[1, 1, 1, 1]
            end

            @testset "2d non-contiguous" begin
                nparr = pycall(np."ndarray", PyObject,
                                buffer=UInt32[1,0,2,0,1,0,2,0,
                                              1,0,2,0,1,0,2,0], order="f",
                                dtype="i4", shape=(2, 4), strides=(8,16))
                pyarr = PyArray(nparr)

                # The convert goes via a PyArray then a `copy`
                @test convert(PyAny, nparr) == [1 1 1 1; 2 2 2 2]
                pyarr = convert(PyArray, nparr)
                @test eltype(pyarr) == Int32
                @test pyarr.info.st == (8, 16)
                # not f_contig because not contiguous
                @test pyarr.f_contig == false
                @test copy(pyarr) == Int32[1 1 1 1; 2 2 2 2]
            end
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
            @test nca[3] == get(ao,2)
            @test nca[4] == get(ao,3)
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
            @test nca[3,2] == get(ao, (2,1))
            @test nca[2,3] == get(ao, (1,2))
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
            @test nca[2,3,4] == get(ao, (1,2,3))
            @test nca[3,2,4] == get(ao, (2,1,3))
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

        @testset "bounds checks" begin
            a = PyArray(pytestarray(3,4,1))
            @test a[2,3,1] == a[2,3] == a[2,3,1,1,1] == a[8]
            @test_throws BoundsError a[5,3,1]
            @test_throws BoundsError a[2,6,1]
            @test_throws BoundsError a[2,3,3]
            @test_throws BoundsError a[2,3,1,2]
            @test_throws BoundsError PyArray(pytestarray(3,4,2))[2,3]
        end

        @testset "similar on PyArray PyVec getindex" begin
            jlarr1 = [1:10;]
            jlarr2 = hcat([1:10;], [1:10;])
            pyarr1 = pycall(np."array", PyArray, jlarr1)
            pyarr2 = pycall(np."array", PyArray, jlarr2)
            @test all(pyarr1[1:10]    .== jlarr1[1:10])
            @test all(pyarr2[1:10, 2] .== jlarr2[1:10, 2])
            @test all(pyarr2[1:10, 1:2] .== jlarr2)
        end
    end

    # ctypes.c_void_p supports the buffer protocol as a 0-dimensional array
    let p = convert(Ptr{Cvoid}, 12345)
        @test PyArray(PyObject(p))[] == p
    end
end
