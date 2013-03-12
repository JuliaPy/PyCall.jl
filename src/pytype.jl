# Defining new Python types from Julia (ugly simulation of C headers)

################################################################
# mirror of Python API types and constants from methodobject.h

type PyMethodDef
    ml_name::Ptr{Uint8}
    ml_meth::Ptr{Void}
    ml_flags::Cint
    ml_doc::Ptr{Uint8} # may be NULL
end

# A PyCFunction is a C function of the form
#     PyObject *func(PyObject *self, PyObject *args)
# The first parameter is the "self" function for method, or 
# for module functions it is the module object.  The second
# parameter is either a tuple of args (for METH_VARARGS),
# a single arg (for METH_O), or NULL (for METH_NOARGS).  func
# must return non-NULL (Py_None is okay) unless there was an
# error, in which case an exception must have been set.

# ml_flags should be one of:
const METH_VARARGS = 0x0001 # args are a tuple of arguments
const METH_NOARGS = 0x0004  # no arguments (NULL argument pointer)
const METH_O = 0x0008       # single argument (not wrapped in tuple)

# not sure when these are needed:
const METH_CLASS = 0x0010 # for class methods
const METH_STATIC = 0x0020 # for static methods

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

type PyTypeObject
    # PyObject_HEAD (for non-Py_TRACE_REFS build):
    ob_refcnt::Int
    ob_type::PyPtr
    ob_size::Int # PyObject_VAR_HEAD

    # PyTypeObject fields:
    tp_name::Ptr{Uint8} # required, should be in format "<module>.<name>"

    # warning: these two were Cint for Python <= 2.4
    tp_basicsize::Int # required, = sizeof(instance)
    tp_itemsize::Int

    tp_dealloc::Ptr{Void}
    tp_print::Ptr{Void}
    tp_getattr::Ptr{Void}
    tp_setattr::Ptr{Void}
    tp_compare::Ptr{Void}
    tp_repr::Ptr{Void}

    tp_as_number::Ptr{Void}
    tp_as_sequence::Ptr{Void}
    tp_as_mapping::Ptr{Void}

    tp_hash::Ptr{Void}
    tp_call::Ptr{Void}
    tp_str::Ptr{Void}
    tp_getattro::Ptr{Void}
    tp_setattro::Ptr{Void}

    tp_as_buffer::Ptr{Void}

    tp_flags::Clong # Required, should default to Py_TPFLAGS_DEFAULT

    tp_doc::Ptr{Uint8} # normally set in example code, but may be NULL

    tp_traverse::Ptr{Void}

    tp_clear::Ptr{Void}

    tp_richcompare::Ptr{Void}

    tp_weaklistoffset::Int

    # added in Python 2.2:
    tp_iter::Ptr{Void}
    tp_iternext::Ptr{Void}

    tp_methods::Ptr{Void}
    tp_members::Ptr{Void}
    tp_getset::Ptr{Void}
    tp_base::Ptr{Void}

    tp_dict::PyPtr
    tp_descr_get::Ptr{Void}
    tp_descr_set::Ptr{Void}
    tp_dictoffset::Int

    tp_init::Ptr{Void}
    tp_alloc::Ptr{Void}
    tp_new::Ptr{Void}
    tp_free::Ptr{Void}
    tp_is_gc::Ptr{Void}

    tp_bases::PyPtr
    tp_mro::PyPtr
    tp_cache::PyPtr
    tp_subclasses::PyPtr
    tp_weaklist::PyPtr
    tp_del::Ptr{Void}

    # added in Python 2.6:
    tp_version_tag::Cuint

    # only used for COUNT_ALLOCS builds of Python
    tp_allocs::Int
    tp_frees::Int
    tp_maxalloc::Int
    tp_prev::Ptr{Void}
    tp_next::Ptr{Void}

    # Julia-specific fields, after the end of the Python structure:

    # save the tp_name Julia string so that it is not garbage-collected
    tp_name_save::ASCIIString

    function PyTypeObject(name::String, basicsize::Integer, init::Function)
        # (Note: don't worry about caching things like pyversion checks
        #  since new PyTypeObjects are created only infrequently)
        vers = pyversion()
        if WORD_SIZE == 64 && vers <= v"2.4"
            error("requires Python 2.5 or later on 64-bit systems")
        end
        if dlsym_e(libpython::Ptr{Void}, :_Py_NewReference) != C_NULL
            # when Python is compiled with Py_TRACE_REFS, _Py_NewReference
            # becomes a function (otherwise it is a macro), which allows
            # us to detect debugging builds.  These are not supported
            # here because it adds two extra PyObject* fields to the
            # beginning of PyObject_HEAD, requiring one to modify the
            # structures above.  (In theory, we could do this at runtime
            # but it doesn't seem worth it if Python debug builds are rare)
            error("Python debug builds (Py_TRACE_REFS) are not supported")
        end
        # figure out Py_TPFLAGS_DEFAULT, depending on Python version
        Py_TPFLAGS_HAVE_STACKLESS_EXTENSION = try pyimport("stackless")
            Py_TPFLAGS_HAVE_STACKLESS_EXTENSION_; catch 0; end
        Py_TPFLAGS_DEFAULT = 
          vers >= v"3.0" ? (Py_TPFLAGS_HAVE_STACKLESS_EXTENSION |
                            Py_TPFLAGS_HAVE_VERSION_TAG) :
          vers >= v"2.5" ? (Py_TPFLAGS_HAVE_GETCHARBUFFER |
                            Py_TPFLAGS_HAVE_SEQUENCE_IN |
                            Py_TPFLAGS_HAVE_INPLACEOPS |
                            Py_TPFLAGS_HAVE_RICHCOMPARE |
                            Py_TPFLAGS_HAVE_WEAKREFS |
                            Py_TPFLAGS_HAVE_ITER |
                            Py_TPFLAGS_HAVE_CLASS |
                            Py_TPFLAGS_HAVE_STACKLESS_EXTENSION |
                            Py_TPFLAGS_HAVE_INDEX) :
          error("Python < 2.5 not supported")
        name_save = bytestring(name)
        t = new(0,C_NULL,0,
                convert(Ptr{Uint8}, name_save),
                convert(Int, basicsize), 0,
                C_NULL,C_NULL,C_NULL,C_NULL,C_NULL,C_NULL, # tp_dealloc ...
                C_NULL,C_NULL,C_NULL, # tp_as_number...
                C_NULL,C_NULL,C_NULL,C_NULL,C_NULL, # tp_hash ...
                C_NULL, # tp_as_buffer
                Py_TPFLAGS_DEFAULT,
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
                name_save)
        init(t) # initialize any other fields as needed
        if t.tp_new == C_NULL
            t.tp_new = @pysym :PyType_GenericNew
        end
        @pycheckzi ccall((@pysym :PyType_Ready), Cint, (Ptr{PyTypeObject},), &t)
        ccall((@pysym :Py_IncRef), Void, (Ptr{PyTypeObject},), &t)
        return t
    end
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
end

################################################################
# from Python structmember.h:

# declare immutable because we need a C-like array of these
immutable PyMemberDef
    name::Ptr{Uint8}
    typ::Cint
    offset::Int # warning: was Cint for Python <= 2.4
    flags::Cint
    doc::Ptr{Uint8}
    PyMemberDef(name,typ,offset,flags,doc) =
        new(convert(Ptr{Uint8},name),
            convert(Cint,typ),
            convert(Int,offset),
            convert(Cint,flags),
            convert(Ptr{Uint8},doc))
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
# Wrap a Python type around a Julia Any object

# the PyMemberDef array must not be garbage-collected
const pyjlwrap_membername = bytestring("jl_value")
const pyjlwrap_doc = bytestring("Julia jl_value_t* (Any object)")
const pyjlwrap_members = 
  PyMemberDef[ PyMemberDef(pyjlwrap_membername,
                           T_PYSSIZET, sizeof_PyObject_HEAD, READONLY,
                           pyjlwrap_doc),
               PyMemberDef(C_NULL,0,0,0,C_NULL) ]

immutable Py_jlWrap
    # PyObject_HEAD (for non-Py_TRACE_REFS build):
    ob_refcnt::Int
    ob_type::PyPtr

    jl_value::Any
end

# destructor for jlwrap instance, assuming it was created with pyjlwrap_new
function pyjlwrap_dealloc(o::PyPtr)
    global pycall_gc
    try
        delete!(pycall_gc::Dict{PyPtr,Any}, o)
        # not sure what to do if there is an exception here 
    end
    return nothing
end

unsafe_pyjlwrap_to_objref(o::PyPtr) = 
  unsafe_pointer_to_objref(unsafe_ref(convert(Ptr{Ptr{Void}}, o), 3))

function pyjlwrap_repr(o::PyPtr)
    o = PyObject(try string("<PyCall.jlwrap ",unsafe_pyjlwrap_to_objref(o),">")
                 catch "<PyCall.jlwrap NULL>"; end)
    oret = o.o
    o.o = C_NULL # don't decref
    return oret
end

function pyjlwrap_hash(o::PyPtr) 
    h = hash(unsafe_pyjlwrap_to_objref(o))
    # Python hashes are not permitted to return -1!!
    return h == uint(-1) ? pysalt::Uint : h::Uint
end

# 32-bit hash on 64-bit machines, needed for Python < 3.2 with Windows
function pyjlwrap_hash32(o::PyPtr)
    h = ccall(:int64to32hash, Uint32, (Uint64,), 
              hash(unsafe_pyjlwrap_to_objref(o)))
    # Python hashes are not permitted to return -1!!
    return h == uint32(-1) ? uint32(pysalt::Uint) : h::Uint32
end

jlWrapType = PyTypeObject()

function pyjlwrap_init()
    if pyversion() < v"2.6"
        error("Python version 2.6 or later required for T_PYSSIZET")
    end
    global jlWrapType
    if (jlWrapType::PyTypeObject).tp_name == C_NULL
        jlWrapType::PyTypeObject =
          PyTypeObject("PyCall.jlwrap", sizeof(Py_jlWrap),
                       t::PyTypeObject -> begin
                           t.tp_flags |= Py_TPFLAGS_BASETYPE
                           t.tp_members = convert(Ptr{Void}, pyjlwrap_members);
                           t.tp_dealloc = cfunction(pyjlwrap_dealloc,
                                                    Void, (PyPtr,))
                           t.tp_repr = cfunction(pyjlwrap_repr,
                                                 PyPtr, (PyPtr,))
                           t.tp_hash = pyhashlong::Bool && WORD_SIZE == 64 &&
                                       sizeof(Clong) == 4 ?
                              cfunction(pyjlwrap_hash32, Uint32, (PyPtr,)) :
                              cfunction(pyjlwrap_hash, Uint, (PyPtr,))
                       end)
    end

end

# use this to create a new jlwrap type, with init to set up custom members
function pyjlwrap_type(name::String, init::Function)
    pyjlwrap_init()
    PyTypeObject(name, 
                 sizeof(Py_jlWrap) + sizeof(PyPtr), # must be > base type
                 t::PyTypeObject -> begin
                     t.tp_base = ccall(:jl_value_ptr, Ptr{Void}, 
                                       (Ptr{PyTypeObject},),
                                       &(jlWrapType::PyTypeObject))
                     init(t)
                 end)
end

# Given a jlwrap type, create a new instance (and save value for gc)
# (Careful: not sure if this works if value is an isbits type,
#  since, the jl_value_t* may be to a temporary copy.  But don't need
#  to wrap isbits types in Python objects anyway.)
function pyjlwrap_new(pyT::PyTypeObject, value::Any)
    global pycall_gc
    o = PyObject(@pycheckn ccall((@pysym :_PyObject_New),
                                 PyPtr, (Ptr{PyTypeObject},), &pyT))
    (pycall_gc::Dict{PyPtr,Any})[o.o] = value
    p = convert(Ptr{Ptr{Void}}, o.o)
    unsafe_assign(p, ccall(:jl_value_ptr, Ptr{Void}, (Any,), value), 3)
    return o
end

is_pyjlwrap(o::PyObject) = (jlWrapType::PyTypeObject).tp_name != C_NULL && ccall((@pysym :PyObject_IsInstance), Cint, (PyPtr,Ptr{PyTypeObject}), o, &(jlWrapType::PyTypeObject)) == 1

################################################################
# Fallback conversion: if we don't have a better conversion function,
# just wrap the Julia object in a Python object

PyObject(x::Any) = begin  pyjlwrap_init(); pyjlwrap_new(jlWrapType, x); end

################################################################

