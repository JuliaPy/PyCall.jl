# Finding python. This is slightly complicated in order to support using PyCall
# from libjulia. We check if python symbols are present in the current process
# and if so do not use the deps.jl file, getting everything we need from the
# current process instead.

# proc_handle = unsafe_load(cglobal(:jl_exe_handle, Ptr{Cvoid}))

include(joinpath(dirname(@__FILE__), "..", "deps", "deps.jl"))

const InProcessLibPyHandle = EnvString("PYCALL_INPROC_LIBPYPTR", "")
const InProcessProcID = EnvString("PYCALL_INPROC_PROCID", "")

mutable struct PointerLoader{P <: Ptr, F <: Function}
    ptr :: P
    ptr_initializer :: F
    function PointerLoader(ptr_initializer::F) where {F <: Function}
        ptr = ptr_initializer()
        new{typeof(ptr), F}(ptr, ptr_initializer)
    end
end

function isloaded(p::PointerLoader)
    p.ptr != C_NULL
end

function _unload(p::PointerLoader{P}) where P
    p.ptr = reinterpret(P, C_NULL)
end

const _start_python_from_julia = Ref(false)

function start_python_from_julia()
    Ptr(LibPyHandle)
    _start_python_from_julia[]
end

const LibPyHandle = PointerLoader(() -> _SetupPythonEnv())

function Core.Ptr(pl::PointerLoader{P}) where P
    if pl.ptr == reinterpret(P, C_NULL)
        pl.ptr = pl.ptr_initializer()
    end
    return pl.ptr
end

const pyversion = vparse(split(Py_GetVersion(Ptr(LibPyHandle)))[1])

# PyUnicode_* may actually be a #define for another symbol, so
# we cache the correct dlsym
const PyUnicode_AsUTF8String =
    findsym(Ptr(LibPyHandle), :PyUnicode_AsUTF8String, :PyUnicodeUCS4_AsUTF8String, :PyUnicodeUCS2_AsUTF8String)
const PyUnicode_DecodeUTF8 =
    findsym(Ptr(LibPyHandle), :PyUnicode_DecodeUTF8, :PyUnicodeUCS4_DecodeUTF8, :PyUnicodeUCS2_DecodeUTF8)

# Python 2/3 compatibility: cache symbols for renamed functions
if hassym(Ptr(LibPyHandle), :PyString_FromStringAndSize)
    const PyString_FromStringAndSize = :PyString_FromStringAndSize
    const PyString_AsStringAndSize = :PyString_AsStringAndSize
    const PyString_Size = :PyString_Size
    const PyString_Type = :PyString_Type
else
    const PyString_FromStringAndSize = :PyBytes_FromStringAndSize
    const PyString_AsStringAndSize = :PyBytes_AsStringAndSize
    const PyString_Size = :PyBytes_Size
    const PyString_Type = :PyBytes_Type
end

# hashes changed from long to intptr_t in Python 3.2
const Py_hash_t = pyversion < v"3.2" ? Clong : Int

# whether to use unicode for strings by default, ala Python 3
const pyunicode_literals = pyversion >= v"3.0"

function pysym_impl(::Val{funcname}) where funcname
    PointerLoader(() ->
        Libdl.dlsym(Ptr(LibPyHandle), funcname))
end

function pyglobal_impl(::Val{name}) where name
    PointerLoader(() ->
        Libdl.dlsym(Ptr(LibPyHandle), name))
end

function pyglobalobj_impl(::Val{name}) where name
    PointerLoader(() ->
        reinterpret(Ptr{PyObject_struct}, Libdl.dlsym(Ptr(LibPyHandle),  name)))
end

function pyglobalobjptr_impl(::Val{name}) where name
    PointerLoader(() ->
        unsafe_load(reinterpret(Ptr{Ptr{PyObject_struct}}, Libdl.dlsym(Ptr(LibPyHandle),  name))))
end

macro pysym(funcname)
    :( $Ptr($pysym_impl($Val($(esc(funcname))))) )
end

macro pyglobal(name)
    :( $Ptr($pyglobal_impl($Val($(esc(name))))) )
end

macro pyglobalobj(name)
    :( $Ptr($pyglobalobj_impl($Val($(esc(name))))) )
end

macro pyglobalobjptr(name)
    :( $Ptr($pyglobalobjptr_impl($Val($(esc(name))))) )
end
