using Base.Test
using PyCall

roundtrip(T, x) = convert(T, PyObject(x))
roundtrip(x) = roundtrip(PyAny, x)
roundtripeq(T, x) = roundtrip(T, x) == x
roundtripeq(x) = roundtrip(x) == x

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

#############################################################
# PyBuffer Tests

@test PyCall.pycheckbuffer(array.array("f", [1.0, 2.0, 3.0])) == true
@test PyCall.pycheckbuffer(np.array([1,2,3])) == true

# Check 1D arrays
a = array.array("f", [1,2,3])
view = PyCall.pygetbuffer(a, PyCall.PyBUF_FULL)
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
view = PyCall.pygetbuffer(a, PyCall.PyBUF_FULL)
@test ndims(view) == 1
@test length(view) == 3
@test sizeof(view) == sizeof(Float64) * 3
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
view = PyCall.pygetbuffer(a, PyCall.PyBUF_FULL)
@test ndims(view) == 2
@test length(view) == 6
@test sizeof(view) == sizeof(Int64) * 6
@test size(view) == (2,3)
@test strides(view) == (3,1)
@test PyCall.aligned(view) == true
@test PyCall.c_contiguous(view) == true
@test PyCall.f_contiguous(view) == false
@test PyCall.pyfmt(view) == "l"

# Check Multi-D C ordered arrays
a = np.ones((10,5,3), dtype="float32", order="C")
@test a[:dtype] == np.dtype("float32")
view = PyCall.pygetbuffer(a, PyCall.PyBUF_FULL)
@test ndims(view) == 3
@test length(view) == (10 * 5 * 3)
@test sizeof(view) == sizeof(Float32) * (10 * 5 * 3)
@test size(view) == (10, 5, 3)
@test strides(view) == (5 * 3, 3, 1)
@test PyCall.aligned(view) == true
@test PyCall.c_contiguous(view) == true
@test PyCall.f_contiguous(view) == false
@test PyCall.pyfmt(view) == "f"

# Check Multi-D F ordered arrays
a = np.ones((10,5,3), dtype="uint8", order="F")
@test a[:dtype] == np.dtype("uint8")
view = PyCall.pygetbuffer(a, PyCall.PyBUF_FULL)
@test ndims(view) == 3
@test length(view) == (10 * 5 * 3)
@test sizeof(view) == sizeof(Uint8) * (10 * 5 * 3)
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
aa = np.frombuffer(a)
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
aa = np.frombuffer(a)
@test np.shape(aa) == size(a)

npa = np.ones((10,10,20), dtype="float32")
a = PyArray(npa)
@test size(a) == (10,10,20)
@test strides(a) == (20 * 10, 20, 1)
@test ndims(a) == 3
@test size(similar(a)) == size(a)
@test eltype(similar(a)) == eltype(a)
@test summary(a) == "10x10x20 Float32 PyArray"

aa = np.frombuffer(a)
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
