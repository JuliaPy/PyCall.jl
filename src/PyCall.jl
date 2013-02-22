module PyCall

export pyinitialize, pyfinalize, pycall, pyimport, pybuiltin, PyObject,
       pyfunc, PyPtr, pyincref, pydecref, pyversion, PyArray, PyArray_Info,
       pyerr_check, pyerr_clear, pytype_query, PyAny, @pyimport, PyWrapper,
       PyDict, pyisinstance, pywrap

import Base.size, Base.ndims, Base.similar, Base.copy, Base.ref, Base.assign,
       Base.stride, Base.convert, Base.pointer, Base.summary, Base.convert,
       Base.show, Base.has, Base.keys, Base.values, Base.eltype, Base.get,
       Base.delete!, Base.empty!, Base.length, Base.isempty, Base.start,
       Base.done, Base.next, Base.filter!

typealias PyPtr Ptr{Void} # type for PythonObject* in ccall

#########################################################################

# Global configuration variables.  Note that, since Julia does not allow us
# to attach type annotations to globals, we need to annotate these explicitly
# as initialized::Bool and libpython::Ptr{Void} when we use them.
initialized = false # whether Python is initialized
libpython = C_NULL # Python shared library (from dlopen)

pyfunc(func::Symbol) = dlsym(libpython::Ptr{Void}, func)

# Macro version of pyinitialize() to inline initialized? check
macro pyinitialize()
    :(initialized::Bool ? nothing : pyinitialize())
end

#########################################################################
# Wrapper around Python's C PyObject* type, with hooks to Python reference
# counting and conversion routines to/from C and Julia types.

type PyObject
    o::PyPtr # the actual PyObject*

    # For copy-free wrapping of arrays, the PyObject may reference a
    # pointer to Julia array data.  In this case, we must keep a
    # reference to the Julia object inside the PyObject so that the
    # former is not garbage collected before the latter.
    keep::Any

    function PyObject(o::PyPtr, keep::Any)
        po = new(o, keep)
        finalizer(po, pydecref)
        return po
    end
end

PyObject(o::PyPtr) = PyObject(o, nothing) # no Julia object to keep

function pydecref(o::PyObject)
    ccall(pyfunc(:Py_DecRef), Void, (PyPtr,), o.o)
    o.o = C_NULL
    o
end

function pyincref(o::PyObject)
    ccall(pyfunc(:Py_IncRef), Void, (PyPtr,), o)
    o
end

pyisinstance(o::PyObject, t::PyObject) = 
  t.o != C_NULL && ccall(pyfunc(:PyObject_IsInstance), Int32, (PyPtr,PyPtr), o, t.o) == 1

pyisinstance(o::PyObject, t::Symbol) = 
  ccall(pyfunc(:PyObject_IsInstance), Int32, (PyPtr,PyPtr), o, pyfunc(t)) == 1

pyquery(q::Symbol, o::PyObject) =
  ccall(pyfunc(q), Int32, (PyPtr,), o) == 1

# conversion to pass PyObject as ccall arguments:
convert(::Type{PyPtr}, po::PyObject) = po.o

# use constructor for generic conversions to PyObject
convert(::Type{PyObject}, o) = PyObject(o)
PyObject(o::PyObject) = o

#########################################################################

inspect = PyObject(C_NULL) # inspect module, needed for module introspection

# the C API doesn't seem to provide this, so we get it from Python & cache it:
BuiltinFunctionType = PyObject(C_NULL)

# special function type used in NumPy and SciPy (if available)
ufuncType = PyObject(C_NULL)

libpython_name(python::String) =
  replace(readall(`$python -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('LDLIBRARY')"`),
          r"\.so(\.[0-9\.]+)?\s*$|\.dll\s*$", "", 1)

# initialize the Python interpreter (no-op on subsequent calls)
function pyinitialize(python::String)
    global initialized
    global libpython
    global inspect
    global BuiltinFunctionType
    global ufuncType
    if (!initialized::Bool)
        if isdefined(:dlopen_global) # see Julia issue #2317
            libpython::Ptr{Void} = dlopen_global(libpython_name(python))
        elseif method_exists(dlopen,(String,Integer))
            libpython::Ptr{Void} = dlopen(libpython_name(python),
                                          RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        else # Julia 0.1 - can't support inter-library dependencies
            libpython::Ptr{Void} = dlopen(libpython_name(python))
        end
        ccall(pyfunc(:Py_SetProgramName), Void, (Ptr{Uint8},), 
              bytestring(python))
        ccall(pyfunc(:Py_InitializeEx), Void, (Int32,), 0)
        initialized::Bool = true
        inspect::PyObject = pyimport("inspect")
        BuiltinFunctionType::PyObject = pyimport("types")["BuiltinFunctionType"]
        try
            ufuncType = pyimport("numpy")["ufunc"]
        catch
            ufuncType = PyObject(C_NULL) # NumPy not available
        end
    end
    return
end

pyinitialize() = pyinitialize("python") # default Python executable name
libpython_name() = libpython_name("python") # ditto

# end the Python interpreter and free associated memory
function pyfinalize()
    global initialized
    global libpython
    global inspect
    global BuiltinFunctionType
    if (initialized::Bool)
        npyfinalize()
        pydecref(ufuncType)
        pydecref(BuiltinFunctionType::PyObject)
        pydecref(inspect::PyObject)
        gc() # collect/decref any remaining PyObjects
        ccall(pyfunc(:Py_Finalize), Void, ())
        dlclose(libpython::Ptr{Void})
        libpython::Ptr{Void} = C_NULL
        initialized::Bool = false
    end
    return
end

# Return the Python version as a Julia VersionNumber<
pyversion() = VersionNumber(convert((Int,Int,Int,String,Int), 
                                    pyimport("sys")["version_info"])[1:3]...)

#########################################################################
# Conversion of Python exceptions into Julia exceptions

# call when we are throwing our own exception
pyerr_clear() = ccall(pyfunc(:PyErr_Clear), Void, ())

type PyError <: Exception
    msg::String
    o::PyObject
end

function pyerr_check(msg::String, val::Any)
    # note: don't call pyinitialize here since we will
    # only use this in contexts where initialization was already done
    e = ccall(pyfunc(:PyErr_Occurred), PyPtr, ())
    if e != C_NULL
        o = pyincref(PyObject(e)) # PyErr_Occurred returns borrowed ref
        pyerr_clear()
        throw(PyError(msg, o))
    end
    val # the val argument is there just to pass through to the return value
end

pyerr_check(msg::String) = pyerr_check(msg, nothing)
pyerr_check() = pyerr_check("")

# Macros for common pyerr_check("Foo", ccall(pyfunc(:Foo), ...)) pattern.
# (The "i" variant assumes Python is initialized.)
macro pychecki(ex)
    :(pyerr_check($(string(ex.args[1].args[2])), $ex))
end
macro pycheck(ex)
    quote
        @pyinitialize
        @pychecki $ex
    end
end

# Macros to check that ccall(pyfunc(:Foo), ...) returns value != bad
# (The "i" variants assume Python is initialized.)
macro pycheckvi(ex, bad)
    quote
        val = $ex
        if val == $bad
            # throw a PyError if available, otherwise throw ErrorException
            pyerr_check($(string(ex.args[1].args[2])), nothing)
            error($(string(ex.args[1].args[2])), " failed")
        end
        val
    end
end
macro pycheckni(ex)
    :(@pycheckvi $ex C_NULL)
end
macro pycheckzi(ex)
    :(@pycheckvi $ex -1)
end
macro pycheckv(ex, bad)
    quote
        @pyinitialize
        @pycheckvi $ex $bad
    end
end
macro pycheckn(ex)
    quote
        @pyinitialize
        @pycheckni $ex
    end
end
macro pycheckz(ex)
    quote
        @pyinitialize
        @pycheckzi $ex
    end
end

#########################################################################

include("conversions.jl")

#########################################################################
# Pretty-printing PyObject

function show(io::IO, o::PyObject)
    if o.o == C_NULL
        print(io, "PyObject NULL")
    else
        s = ccall(pyfunc(:PyObject_Str), PyPtr, (PyPtr,), o)
        if (s == C_NULL)
            pyerr_clear()
            s = ccall(pyfunc(:PyObject_Repr), PyPtr, (PyPtr,), o)
            if (s == C_NULL)
                pyerr_clear()
                return print(io, "PyObject $(o.o)")
            end
        end
        print(io, "PyObject $(convert(String, PyObject(s)))")
    end
end

#########################################################################
# For o::PyObject, make o["foo"] and o[:foo] equivalent to o.foo in Python

function ref(o::PyObject, s::String)
    if (o.o == C_NULL)
        throw(ArgumentError("ref of NULL PyObject"))
    end
    p = ccall(pyfunc(:PyObject_GetAttrString), PyPtr,
              (PyPtr, Ptr{Uint8}), o, bytestring(s))
    if p == C_NULL
        pyerr_clear()
        throw(KeyError(s))
    end
    return PyObject(p)
end

ref(o::PyObject, s::Symbol) = ref(o, string(s))

#########################################################################
# Create anonymous composite w = pywrap(o) wrapping the object o
# and providing access to o's members (converted to PyAny) as w.member.

abstract PyWrapper

# still provide w[:foo] low-level access to unconverted members:
ref(w::PyWrapper, s) = ref(w.___jl_PyCall_PyObject___, s)

typesymbol(T::AbstractKind) = T.name.name
typesymbol(T::BitsKind) = T.name.name
typesymbol(T::CompositeKind) = T.name.name
typesymbol(T) = :Any # punt

# hack: because of Julia issue #2386, we need to cache pywrap's
#       local variables in globals
pywrap_members = PyObject(C_NULL)
pywrap_o = PyObject(C_NULL)
function pywrap(o::PyObject)
    @pyinitialize
    members = convert(Vector{(String,PyObject)}, 
                      pycall(inspect["getmembers"], PyObject, o))
    tname = gensym("PyCall_PyWrapper")
    global pywrap_members
    global pywrap_o
    pywrap_members::PyObject = members
    pywrap_o::PyObject = o
    @eval begin
        $(expr(:type, expr(:<:, tname, :PyWrapper),
               expr(:block, :(___jl_PyCall_PyObject___::PyObject),
                    map(m -> expr(:(::), symbol(m[1]),
                                  typesymbol(pytype_query(m[2]))), 
                        members)...)))
        $(expr(:call, tname, :pywrap_o,
               [ :(convert(PyAny, pywrap_members[$i][2]))
                for i = 1:length(members) ]...))
    end
end

#########################################################################

pyimport(name::String) =
    PyObject(@pycheckn ccall(pyfunc(:PyImport_ImportModule), PyPtr,
                             (Ptr{Uint8},), bytestring(name)))

pyimport(name::Symbol) = pyimport(string(name))

# convert expressions like :math or :(scipy.special) into module name strings
modulename(s::Symbol) = string(s)
function modulename(e::Expr)
    if e.head == :.
        string(modulename(e.args[1]), :., modulename(e.args[2]))
    elseif e.head == :quote
        modulename(e.args...)
    else
        throw(ArgumentError("invalid module"))
    end
end

macro pyimport(name, optional_varname...)
    mname = modulename(name)
    len = length(optional_varname)
    Name = len > 0 && (len != 2 || optional_varname[1] != :as) ? 
      throw(ArgumentError("usage @pyimport module [as name]")) :
      (len == 2 ? optional_varname[2] :
       typeof(name) == Symbol ? name :
       throw(ArgumentError("$mname is not a valid module variable name, use @pyimport $mname as <name>")))
    quote
        $(esc(Name)) = pywrap(pyimport($mname))
        nothing
    end
end

#########################################################################

# look up a global variable (in module __main__)
function pybuiltin(name::String)
    main = @pycheckn ccall(pyfunc(:PyImport_AddModule), 
                           PyPtr, (Ptr{Uint8},),
                           bytestring("__main__"))
    PyObject(@pycheckni ccall(pyfunc(:PyObject_GetAttrString), PyPtr,
                              (PyPtr, Ptr{Uint8}), main,
                              bytestring("__builtins__")))[bytestring(name)]
end

pybuiltin(name::Symbol) = pybuiltin(string(name))

#########################################################################

function pycall(o::PyObject, returntype::Union(Type,NTuple{Type}), args...)
    oargs = map(PyObject, args)
    # would rather call PyTuple_Pack, but calling varargs functions
    # with ccall and argument splicing seems problematic right now.
    arg = PyObject(@pycheckn ccall(pyfunc(:PyTuple_New), PyPtr, (Int,), 
                                   length(args)))
    for i = 1:length(args)
        @pycheckzi ccall(pyfunc(:PyTuple_SetItem), Int32, (PyPtr,Int,PyPtr),
                         arg, i-1, oargs[i])
        pyincref(oargs[i]) # PyTuple_SetItem steals the reference
    end
    ret = PyObject(@pycheckni ccall(pyfunc(:PyObject_CallObject), PyPtr,
                                    (PyPtr,PyPtr), o, arg))
    jret = convert(returntype, ret)
    return jret
end

#########################################################################

end # module PyCall
