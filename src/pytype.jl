# Defining new Python types from Julia (ugly simulation of C headers)

################################################################
# Python expects the PyMethodDef and similar strings to be constants,
# so we define anonymous globals to hold them, returning the pointer
const permanent_strings = String[]
function gstring_ptr(name::AbstractString, s::AbstractString)
    g = String(s)
    push!(permanent_strings, g)
    unsafe_convert(Ptr{UInt8}, g)
end

################################################################
# mirror of Python API types and constants from methodobject.h

struct PyMethodDef
    ml_name::Ptr{UInt8}
    ml_meth::Ptr{Cvoid}
    ml_flags::Cint
    ml_doc::Ptr{UInt8} # may be NULL
end

# A PyCFunction is a C function of the form
#     PyObject *func(PyObject *self, PyObject *args)
# or
#     PyObject *func(PyObject *self, PyObject *args, PyObject *kwargs)
# The first parameter is the "self" function for method, or
# for module functions it is the module object.  The second
# parameter is either a tuple of args (for METH_VARARGS),
# a single arg (for METH_O), or NULL (for METH_NOARGS).  func
# must return non-NULL (Py_None is okay) unless there was an
# error, in which case an exception must have been set.

# ml_flags should be one of:
const METH_VARARGS = 0x0001 # args are a tuple of arguments
const METH_KEYWORDS = 0x0002  # two arguments: the varargs and the kwargs
const METH_NOARGS = 0x0004  # no arguments (NULL argument pointer)
const METH_O = 0x0008       # single argument (not wrapped in tuple)

# not sure when these are needed:
const METH_CLASS = 0x0010 # for class methods
const METH_STATIC = 0x0020 # for static methods

const NULL_UInt8_Ptr = convert(Ptr{UInt8}, C_NULL)
function PyMethodDef(name::AbstractString, meth::Ptr{Cvoid}, flags::Integer, doc::AbstractString="")
    PyMethodDef(gstring_ptr(name, name),
                meth,
                convert(Cint, flags),
                isempty(doc) ? NULL_UInt8_Ptr : gstring_ptr(name, doc))
end

# used as sentinel value to end method arrays:
PyMethodDef() = PyMethodDef(NULL_UInt8_Ptr, C_NULL,
                            convert(Cint, 0), NULL_UInt8_Ptr)

################################################################
# mirror of Python API types and constants from descrobject.h

struct PyGetSetDef
    name::Ptr{UInt8}
    get::Ptr{Cvoid}
    set::Ptr{Cvoid} # may be NULL for read-only members
    doc::Ptr{UInt8} # may be NULL
    closure::Ptr{Cvoid} # pass-through thunk, may be NULL
end

# probably should be changed to macro to avoid interpolating into @cfunction:
# (commented out for now since we aren't actually using it)
# function PyGetSetDef(name::AbstractString, get::Function,set::Function, doc::AbstractString="")
#     PyGetSetDef(gstring_ptr(name, name),
#                 @cfunction($get, PyPtr, (PyPtr,Ptr{Cvoid})),
#                 @cfunction($set, Int, (PyPtr,PyPtr,Ptr{Cvoid})),
#                 isempty(doc) ? NULL_UInt8_Ptr : gstring_ptr(name, doc),
#                 C_NULL)
# end

# used as sentinel value to end attribute arrays:
PyGetSetDef() = PyGetSetDef(NULL_UInt8_Ptr, C_NULL, C_NULL, NULL_UInt8_Ptr, C_NULL)

################################################################
# from Python structmember.h:

# declare immutable because we need a C-like array of these
struct PyMemberDef
    name::Ptr{UInt8}
    typ::Cint
    offset::Int # warning: was Cint for Python <= 2.4
    flags::Cint
    doc::Ptr{UInt8}
    PyMemberDef(name,typ,offset,flags,doc) =
        new(unsafe_convert(Ptr{UInt8},name),
            convert(Cint,typ),
            convert(Int,offset),
            convert(Cint,flags),
            unsafe_convert(Ptr{UInt8},doc))
end

# types:
const T_SHORT        =0
const T_INT          =1
const T_LONG         =2
const T_FLOAT        =3
const T_DOUBLE       =4
const T_STRING       =5
const T_OBJECT       =6
const T_CHAR         =7
const T_BYTE         =8
const T_UBYTE        =9
const T_USHORT       =10
const T_UINT         =11
const T_ULONG        =12
const T_STRING_INPLACE       =13
const T_BOOL         =14
const T_OBJECT_EX    =16
const T_LONGLONG     =17 # added in Python 2.5
const T_ULONGLONG    =18 # added in Python 2.5
const T_PYSSIZET     =19 # added in Python 2.6
const T_NONE         =20 # added in Python 3.0

# flags:
const READONLY = 1
const READ_RESTRICTED = 2
const PY_WRITE_RESTRICTED = 4
const RESTRICTED = (READ_RESTRICTED | PY_WRITE_RESTRICTED)

################################################################
# type-flag constants, from Python object.h:

# Python 2.7
const Py_TPFLAGS_HAVE_GETCHARBUFFER  = (0x00000001<<0)
const Py_TPFLAGS_HAVE_SEQUENCE_IN = (0x00000001<<1)
const Py_TPFLAGS_GC = 0 # was sometimes (0x00000001<<2) in Python <= 2.1
const Py_TPFLAGS_HAVE_INPLACEOPS = (0x00000001<<3)
const Py_TPFLAGS_CHECKTYPES = (0x00000001<<4)
const Py_TPFLAGS_HAVE_RICHCOMPARE = (0x00000001<<5)
const Py_TPFLAGS_HAVE_WEAKREFS = (0x00000001<<6)
const Py_TPFLAGS_HAVE_ITER = (0x00000001<<7)
const Py_TPFLAGS_HAVE_CLASS = (0x00000001<<8)
const Py_TPFLAGS_HAVE_INDEX = (0x00000001<<17)
const Py_TPFLAGS_HAVE_NEWBUFFER = (0x00000001<<21)
const Py_TPFLAGS_STRING_SUBCLASS       = (0x00000001<<27)

# Python 3.0+ has only these:
const Py_TPFLAGS_HEAPTYPE = (0x00000001<<9)
const Py_TPFLAGS_BASETYPE = (0x00000001<<10)
const Py_TPFLAGS_READY = (0x00000001<<12)
const Py_TPFLAGS_READYING = (0x00000001<<13)
const Py_TPFLAGS_HAVE_GC = (0x00000001<<14)
const Py_TPFLAGS_HAVE_VERSION_TAG   = (0x00000001<<18)
const Py_TPFLAGS_VALID_VERSION_TAG  = (0x00000001<<19)
const Py_TPFLAGS_IS_ABSTRACT = (0x00000001<<20)
const Py_TPFLAGS_INT_SUBCLASS         = (0x00000001<<23)
const Py_TPFLAGS_LONG_SUBCLASS        = (0x00000001<<24)
const Py_TPFLAGS_LIST_SUBCLASS        = (0x00000001<<25)
const Py_TPFLAGS_TUPLE_SUBCLASS       = (0x00000001<<26)
const Py_TPFLAGS_BYTES_SUBCLASS       = (0x00000001<<27)
const Py_TPFLAGS_UNICODE_SUBCLASS     = (0x00000001<<28)
const Py_TPFLAGS_DICT_SUBCLASS        = (0x00000001<<29)
const Py_TPFLAGS_BASE_EXC_SUBCLASS    = (0x00000001<<30)
const Py_TPFLAGS_TYPE_SUBCLASS        = (0x00000001<<31)

# only use this if we have the stackless extension
const Py_TPFLAGS_HAVE_STACKLESS_EXTENSION_ = (0x00000003<<15)

################################################################
# Mirror of PyTypeObject in Python object.h
#  -- assumes non-debugging Python build (no Py_TRACE_REFS)
#  -- most fields can default to 0 except where noted

const sizeof_PyObject_HEAD = sizeof(Int) + sizeof(PyPtr)

mutable struct PyTypeObject
    # PyObject_HEAD (for non-Py_TRACE_REFS build):
    ob_refcnt::Int
    ob_type::PyPtr
    ob_size::Int # PyObject_VAR_HEAD

    # PyTypeObject fields:
    tp_name::Ptr{UInt8} # required, should be in format "<module>.<name>"

    # warning: these two were Cint for Python <= 2.4
    tp_basicsize::Int # required, = sizeof(instance)
    tp_itemsize::Int

    tp_dealloc::Ptr{Cvoid}
    tp_print::Ptr{Cvoid}
    tp_getattr::Ptr{Cvoid}
    tp_setattr::Ptr{Cvoid}
    tp_compare::Ptr{Cvoid}
    tp_repr::Ptr{Cvoid}

    tp_as_number::Ptr{Cvoid}
    tp_as_sequence::Ptr{Cvoid}
    tp_as_mapping::Ptr{Cvoid}

    tp_hash::Ptr{Cvoid}
    tp_call::Ptr{Cvoid}
    tp_str::Ptr{Cvoid}
    tp_getattro::Ptr{Cvoid}
    tp_setattro::Ptr{Cvoid}

    tp_as_buffer::Ptr{Cvoid}

    tp_flags::Clong # Required, should default to Py_TPFLAGS_DEFAULT

    tp_doc::Ptr{UInt8} # normally set in example code, but may be NULL

    tp_traverse::Ptr{Cvoid}

    tp_clear::Ptr{Cvoid}

    tp_richcompare::Ptr{Cvoid}

    tp_weaklistoffset::Int

    # added in Python 2.2:
    tp_iter::Ptr{Cvoid}
    tp_iternext::Ptr{Cvoid}

    tp_methods::Ptr{PyMethodDef}
    tp_members::Ptr{PyMemberDef}
    tp_getset::Ptr{PyGetSetDef}
    tp_base::Ptr{Cvoid}

    tp_dict::PyPtr
    tp_descr_get::Ptr{Cvoid}
    tp_descr_set::Ptr{Cvoid}
    tp_dictoffset::Int

    tp_init::Ptr{Cvoid}
    tp_alloc::Ptr{Cvoid}
    tp_new::Ptr{Cvoid}
    tp_free::Ptr{Cvoid}
    tp_is_gc::Ptr{Cvoid}

    tp_bases::PyPtr
    tp_mro::PyPtr
    tp_cache::PyPtr
    tp_subclasses::PyPtr
    tp_weaklist::PyPtr
    tp_del::Ptr{Cvoid}

    # added in Python 2.6:
    tp_version_tag::Cuint

    # only used for COUNT_ALLOCS builds of Python
    tp_allocs::Int
    tp_frees::Int
    tp_maxalloc::Int
    tp_prev::Ptr{Cvoid}
    tp_next::Ptr{Cvoid}

    # Julia-specific fields, after the end of the Python structure:

    # save the tp_name Julia string so that it is not garbage-collected
    tp_name_save # This is a gc slot that is never read from

    function PyTypeObject()
        new(0,C_NULL,0,
            C_NULL,
            0, 0,
            C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL, # tp_dealloc ...
            C_NULL,C_NULL,C_NULL, # tp_as_number...
            C_NULL,C_NULL,C_NULL,C_NULL,C_NULL, # tp_hash ...
            C_NULL, # tp_as_buffer
            0,
            C_NULL, # tp_doc
            C_NULL, # tp_traverse,
            C_NULL, # tp_clear
            C_NULL, # tp_richcompare
            0, # tp_weaklistoffset
            C_NULL,C_NULL, # tp_iter, tp_iternext
            C_NULL,C_NULL,C_NULL,C_NULL, # tp_methods...
            C_NULL,C_NULL,C_NULL,0, # tp_dict...
            C_NULL,C_NULL,C_NULL,C_NULL,C_NULL, # tp_init ...
            C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL, # tp_bases...
            0, # tp_version_tag
            0,0,0,C_NULL,C_NULL, # tp_allocs...
            "")
    end
    PyTypeObject(name::AbstractString, basicsize::Integer, init::Function) =
        PyTypeObject!(PyTypeObject(), name, basicsize, init)
end

# Often, PyTypeObject instances are global constants, which we initialize
# to 0 via PyTypeObject() and then initialize at runtime via PyTypeObject!
function PyTypeObject!(init::Function, t::PyTypeObject, name::AbstractString, basicsize::Integer)
    t.tp_basicsize = convert(Int, basicsize)

    # figure out Py_TPFLAGS_DEFAULT, depending on Python version
    t.tp_flags = # Py_TPFLAGS_DEFAULT =
      pyversion.major >= 3 ?
        (Py_TPFLAGS_HAVE_STACKLESS_EXTENSION[] |
         Py_TPFLAGS_HAVE_VERSION_TAG) :
        (Py_TPFLAGS_HAVE_GETCHARBUFFER |
         Py_TPFLAGS_HAVE_SEQUENCE_IN |
         Py_TPFLAGS_HAVE_INPLACEOPS |
         Py_TPFLAGS_HAVE_RICHCOMPARE |
         Py_TPFLAGS_HAVE_WEAKREFS |
         Py_TPFLAGS_HAVE_ITER |
         Py_TPFLAGS_HAVE_CLASS |
         Py_TPFLAGS_HAVE_STACKLESS_EXTENSION[] |
         Py_TPFLAGS_HAVE_INDEX)

    # Emulate the rooting behavior of a ccall:
    name_save = Base.cconvert(Ptr{UInt8}, name)
    t.tp_name_save = name_save
    t.tp_name = unsafe_convert(Ptr{UInt8}, name_save)

    init(t) # initialize any other fields as needed
    if t.tp_new == C_NULL
        t.tp_new = @pyglobal :PyType_GenericNew
    end
    # TODO change to `Ref{PyTypeObject}` when 0.6 is dropped.
    @pycheckz @pyccall(:PyType_Ready, Cint, (Any,), t)
    @pyccall(:Py_IncRef, Cvoid, (Any,), t)
    return t
end

################################################################
# Wrap a Python type around a Julia Any object

struct Py_jlWrap
    # PyObject_HEAD (for non-Py_TRACE_REFS build):
    ob_refcnt::Int
    ob_type::PyPtr

    jl_value::Any
end

# destructor for jlwrap instance, assuming it was created with pyjlwrap_new
function pyjlwrap_dealloc(o::PyPtr)
    delete!(pycall_gc, o)
    return nothing
end

unsafe_pyjlwrap_to_objref(o::PyPtr) =
  unsafe_pointer_to_objref(unsafe_load(convert(Ptr{Ptr{Cvoid}}, o), 3))

pyjlwrap_repr(o::PyPtr) =
    pystealref!(PyObject(o != C_NULL ? string("<PyCall.jlwrap ",unsafe_pyjlwrap_to_objref(o),">")
                                     : "<PyCall.jlwrap NULL>"))

function pyjlwrap_hash(o::PyPtr)
    h = hash(unsafe_pyjlwrap_to_objref(o))
    # Python hashes are not permitted to return -1!!
    return h == reinterpret(UInt, -1) ? pysalt::UInt : h::UInt
end

# 32-bit hash on 64-bit machines, needed for Python < 3.2 with Windows
const pysalt32 = 0xb592cd9b # hash("PyCall") % UInt32
function pyjlwrap_hash32(o::PyPtr)
    h = ccall(:int64to32hash, UInt32, (UInt64,),
              hash(unsafe_pyjlwrap_to_objref(o)))
    # Python hashes are not permitted to return -1!!
    return h == reinterpret(UInt32, Int32(-1)) ? pysalt32 : h::UInt32
end

docstring(x) = string(Docs.doc(x))

# this function emulates standard attributes of Python functions,
# where possible.
function pyjlwrap_getattr(self_::PyPtr, attr__::PyPtr)
    attr_ = PyObject(attr__) # don't need pyincref because of finally clause below
    try
        f = unsafe_pyjlwrap_to_objref(self_)
        attr = convert(String, attr_)
        if attr in ("__name__","func_name")
            return pystealref!(PyObject(string(f)))
        elseif attr in ("__doc__", "func_doc")
            return pystealref!(PyObject(docstring(f)))
        elseif attr in ("__module__","__defaults__","func_defaults","__closure__","func_closure")
            return pystealref!(PyObject(nothing))
        elseif startswith(attr, "__")
            # TODO: handle __code__/func_code (issue #268)
            return @pyccall(:PyObject_GenericGetAttr, PyPtr, (PyPtr,PyPtr), self_, attr__)
        else
            fidx = Base.fieldindex(typeof(f), Symbol(attr), false)
            if fidx != 0
                return pyreturn(getfield(f, fidx))
            else
                return @pyccall(:PyObject_GenericGetAttr, PyPtr, (PyPtr,PyPtr), self_, attr__)
            end
        end
    catch e
        pyraise(e)
    finally
        attr_.o = PyPtr_NULL # don't decref
    end
    return PyPtr_NULL
end

# constant strings (must not be gc'ed) for pyjlwrap_members
const pyjlwrap_membername = "jl_value"
const pyjlwrap_doc = "Julia jl_value_t* (Any object)"
# other pointer-containing constants that need to be initialized at runtime
const pyjlwrap_members = PyMemberDef[]
const jlWrapType = PyTypeObject() # initialized by pyjlwrap_init in __init__
const Py_TPFLAGS_HAVE_STACKLESS_EXTENSION = Ref(0x00000000)

# use this to create a new jlwrap type, with init to set up custom members
function pyjlwrap_type!(init::Function, to::PyTypeObject, name::AbstractString)
    sz = sizeof(Py_jlWrap) + sizeof(PyPtr) # must be > base type
    PyTypeObject!(to, name, sz) do t::PyTypeObject
        # TODO change to `Ref{PyTypeObject}` when 0.6 is dropped.
        t.tp_base = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), jlWrapType)
        @pyccall(:Py_IncRef, Cvoid, (Any,), jlWrapType)
        init(t)
    end
end

pyjlwrap_type(init::Function, name::AbstractString) =
    pyjlwrap_type!(init, PyTypeObject(), name)

# Given a jlwrap type, create a new instance (and save value for gc)
# (Careful: not sure if this works if value is an isbits type,
#  since, the jl_value_t* may be to a temporary copy.  But don't need
#  to wrap isbits types in Python objects anyway.)
function pyjlwrap_new(pyT::PyTypeObject, value::Any)
    # TODO change to `Ref{PyTypeObject}` when 0.6 is dropped.
    o = PyObject(@pycheckn @pyccall(:_PyObject_New,
                                 PyPtr, (Any,), pyT))
    p = convert(Ptr{Ptr{Cvoid}}, o.o)
    if isimmutable(value)
        # It is undefined to call `pointer_from_objref` on immutable objects.
        # The compiler is free to return basically anything since the boxing is not
        # significant at all.
        # Below is a well defined way to get a pointer (`ptr`) and an object that defines
        # the lifetime of the pointer `ref`.
        ref = Ref{Any}(value)
        pycall_gc[o.o] = ref
        ptr = unsafe_load(Ptr{Ptr{Cvoid}}(pointer_from_objref(ref)))
    else
        pycall_gc[o.o] = value
        ptr = pointer_from_objref(value)
    end
    unsafe_store!(p, ptr, 3)
    return o
end

function pyjlwrap_new(x::Any)
    pyjlwrap_new(jlWrapType, x)
end

# TODO change to `Ref{PyTypeObject}` when 0.6 is dropped.
is_pyjlwrap(o::PyObject) = jlWrapType.tp_new != C_NULL && @pyccall(:PyObject_IsInstance, Cint, (PyPtr, Any), o, jlWrapType) == 1

################################################################
# Fallback conversion: if we don't have a better conversion function,
# just wrap the Julia object in a Python object

PyObject(x::Any) = pyjlwrap_new(x)
