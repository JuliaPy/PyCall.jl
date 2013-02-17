# Julia wrappers around the NumPy API, as part of the PyCall package

#########################################################################
# Initialization (UGLY)

# Linking NumPy's C API from Julia requires some serious hackery,
# because NumPy does not export its symbols in the usual way for
# shared libraries.  Instead, it provides a Python variable
# numpy.core.multiarray._ARRAY_API that points to a lookup table of
# pointers to the API functions and global variables.  Moreover, the
# meaning of this table, along with a lot of other important
# constants, is defined in a C header file which changes between NumPy
# versions, so we need to do some regex parsing of this header file in
# order to extract the necessary information to call NumPy.  Ugly and
# mildly insane, but I don't see much alternative (at least to do this
# purely in Julia).
#
# The result of npy_api_initialize, below, is to produce the following
# tables of API pointers:
npy_api = Dict{Symbol, Ptr{Void}}()

# type alias to shorten type decorations (needed for JIT) of our table
typealias Tnpy_api Dict{Symbol, Ptr{Void}}

# need a global to cache pyimport("numpy.core.multiarray"), in order
# to ensure the module is not garbage-collected as long as we are using it
# for the npy_api pointers.
npy_multiarray = PyObject(C_NULL)

npy_initialized = false # global to prevent multiple initializations

function npyinitialize()
    global npy_api
    global npy_initialized
    global npy_multiarray

    if npy_initialized::Bool
        return
    end
    npy_multiarray::PyObject = pyimport("numpy.core.multiarray")
    if pyversion() < VersionNumber(2,7)
        PyArray_API = ccall(pyfunc(:PyCObject_AsVoidPtr), Ptr{Ptr{Void}}, 
                            (PyPtr,), (npy_multiarray::PyObject)["_ARRAY_API"])
    else
        PyArray_API = ccall(pyfunc(:PyCapsule_GetPointer), Ptr{Ptr{Void}}, 
                            (PyPtr,Ptr{Void}), 
                            (npy_multiarray::PyObject)["_ARRAY_API"], C_NULL)
    end

    # directory for numpy include files to parse
    inc = pycall(pyimport("numpy")["get_include"], String)

    # Parse __multiarray_api.h to obtain length and meaning of PyArray_API
    hdrfile = open(joinpath(inc, "numpy", "__multiarray_api.h"))
    hdr = readall(hdrfile);
    close(hdrfile)
    hdr = replace(hdr, r"\\\s*\n", " "); # rm backslashed newlines
    r = r"^#define\s+([A-Za-z]\w*)\s+\(.*\bPyArray_API\s*\[\s*([0-9]+)\s*\]\s*\)\s*$"m # regex to match #define PyFoo (... PyArray_API[nnn])
    PyArray_API_length = 0
    offset = 1
    m = match(r, hdr, offset)
    while m != nothing # search for max index into PyArray_API
        PyArray_API_length = max(PyArray_API_length, int(m.captures[2])+1)
        offset = m.offset + length(m.match)
        m = match(r, hdr, offset)
    end
    API = pointer_to_array(PyArray_API, (PyArray_API_length,))
    offset = 1
    m = match(r, hdr, offset)
    while m != nothing # build npy_api table
        merge!(npy_api::Tnpy_api,
               [ symbol(m.captures[1]) => API[int(m.captures[2])+1] ])
        offset = m.offset + length(m.match)
        m = match(r, hdr, offset)
    end
    if !has(npy_api::Tnpy_api, :PyArray_New)
        error("failure parsing NumPy PyArray_API symbol table")
    end

    npy_initialized::Bool = true
    return
end

function npyfinalize()
    global npy_api
    global npy_initialized
    global npy_multiarray

    if npy_initialized::Bool
        empty!(npy_api::Tnpy_api)
        pydecref(npy_multiarray::PyObject)
        npy_initialized::Bool = false
    end
    return
end

#########################################################################
# Hard-coded constant values, copied from numpy/ndarraytypes.h ...
# the values of these seem to have been stable for some time, and
# the NumPy developers seem to have some awareness of binary compatibility

# NPY_TYPES:
const NPY_BOOL = int32(0)
const NPY_BYTE = int32(1)
const NPY_UBYTE = int32(2)
const NPY_SHORT = int32(3)
const NPY_USHORT = int32(4)
const NPY_INT = int32(5)
const NPY_UINT = int32(6)
const NPY_LONG = int32(7)
const NPY_ULONG = int32(8)
const NPY_LONGLONG = int32(9)
const NPY_ULONGLONG = int32(10)
const NPY_FLOAT = int32(11)
const NPY_DOUBLE = int32(12)
const NPY_LONGDOUBLE = int32(13)
const NPY_CFLOAT = int32(14)
const NPY_CDOUBLE = int32(15)
const NPY_CLONGDOUBLE = int32(16)
const NPY_OBJECT = int32(17)
const NPY_STRING = int32(18)
const NPY_UNICODE = int32(19)
const NPY_VOID = int32(20)

# NPY_ORDER:
const NPY_ANYORDER=int32(-1)
const NPY_CORDER=int32(0)
const NPY_FORTRANORDER=int32(1)

# flags:
const NPY_ARRAY_C_CONTIGUOUS = int32(1)
const NPY_ARRAY_F_CONTIGUOUS = int32(2)
const NPY_ARRAY_ALIGNED = int32(0x0100)
const NPY_ARRAY_WRITEABLE = int32(0x0400)
const NPY_ARRAY_OWNDATA = int32(0x0004)
const NPY_ARRAY_ENSURECOPY = int32(0x0020)
const NPY_ARRAY_ENSUREARRAY = int32(0x0040)
const NPY_ARRAY_FORCECAST = int32(0x0010)
const NPY_ARRAY_UPDATEIFCOPY = int32(0x1000)
const NPY_ARRAY_NOTSWAPPED = int32(0x0200)
const NPY_ARRAY_ELEMENTSTRIDES = int32(0x0080)

#########################################################################
# conversion from Julia types to NPY_TYPES constant

npy_type(::Type{Int8}) = NPY_BYTE
npy_type(::Type{Uint8}) = NPY_UBYTE
npy_type(::Type{Int16}) = NPY_SHORT
npy_type(::Type{Uint16}) = NPY_USHORT
npy_type(::Type{Int32}) = NPY_INT
npy_type(::Type{Uint32}) = NPY_UINT
npy_type(::Type{Int64}) = NPY_LONGLONG
npy_type(::Type{Uint64}) = NPY_ULONGLONG
npy_type(::Type{Float32}) = NPY_FLOAT
npy_type(::Type{Float64}) = NPY_DOUBLE
npy_type(::Type{Complex64}) = NPY_CFLOAT
npy_type(::Type{Complex128}) = NPY_CDOUBLE
npy_type(::Type{PyObject}) = NPY_OBJECT

typealias NPY_TYPES Union(Int8,Uint8,Int16,Uint16,Int32,Uint32,Int64,Uint64,Float32,Float64,Complex64,Complex128,PyObject)

#########################################################################
# copy-less conversion of Julia arrays to NumPy arrays.

function PyObject{T<:NPY_TYPES}(a::StridedArray{T})
    npyinitialize()
    p = ccall(npy_api[:PyArray_New], PyPtr,
              (PyPtr,Int32,Ptr{Int},Int32, Ptr{Int},Ptr{T}, Int32,Int32,PyPtr),
              npy_api[:PyArray_Type], 
              ndims(a), [size(a)...], npy_type(T),
              [strides(a)...] * sizeof(eltype(a)), a, sizeof(eltype(a)),
              NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
              C_NULL)
    return PyObject(p, a)
end

#########################################################################
