module PyCall

export pyinitialize, pyfinalize, pycall, pyimport, pybuiltin, PyObject,
       pyfunc, PyPtr, pyincref, pydecref, pyversion, PyArray, PyArray_Info,
       pyerr_check, pyerr_clear, pytype_query, PyAny, @pyimport, PyWrapper,
       PyDict, pyisinstance, pywrap, @pykw, pytypeof

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
finalized = false # whether Python has been finalized
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
    function PyObject(o::PyPtr)
        po = new(o)
        finalizer(po, pydecref)
        return po
    end
end

function pydecref(o::PyObject)
    if initialized::Bool # don't decref after pyfinalize!
        ccall(pyfunc(:Py_DecRef), Void, (PyPtr,), o.o)
    end
    o.o = C_NULL
    o
end

function pyincref(o::PyObject)
    ccall(pyfunc(:Py_IncRef), Void, (PyPtr,), o)
    o
end

pyisinstance(o::PyObject, t::PyObject) = 
  t.o != C_NULL && ccall(pyfunc(:PyObject_IsInstance), Cint, (PyPtr,PyPtr), o, t.o) == 1

pyisinstance(o::PyObject, t::Symbol) = 
  ccall(pyfunc(:PyObject_IsInstance), Cint, (PyPtr,PyPtr), o, pyfunc(t)) == 1

pyquery(q::Symbol, o::PyObject) =
  ccall(pyfunc(q), Cint, (PyPtr,), o) == 1

pytypeof(o::PyObject) = o.o == C_NULL ? throw(ArgumentError("NULL PyObjects have no Python type")) : pycall(PyCall.TypeType, PyObject, o)

# conversion to pass PyObject as ccall arguments:
convert(::Type{PyPtr}, po::PyObject) = po.o

# use constructor for generic conversions to PyObject
convert(::Type{PyObject}, o) = PyObject(o)
PyObject(o::PyObject) = o

#########################################################################

inspect = PyObject(C_NULL) # inspect module, needed for module introspection

# Python has zillions of types that a function be, in addition to the FunctionType
# in the C API.  We have to obtain these at runtime and cache them in globals
BuiltinFunctionType = PyObject(C_NULL)
TypeType = PyObject(C_NULL) # type constructor
MethodType = PyObject(C_NULL)
MethodWrapperType = PyObject(C_NULL)
# also WrapperDescriptorType = type(list.__add__) and
#      MethodDescriptorType = type(list.append) ... is it worth detecting these?

# special function type used in NumPy and SciPy (if available)
ufuncType = PyObject(C_NULL)

# cache Python None and PyNoneType
pynothing = PyObject(C_NULL)
PyNoneType = PyObject(C_NULL)

# Py_SetProgramName needs its argument to persist as long as Python does
pyprogramname = bytestring("")

# low-level initialization, given a pointer to dlopen result on libpython,
# or C_NULL if python symbols are in the global namespace:
# initialize the Python interpreter (no-op on subsequent calls)
function pyinitialize(libpy::Ptr{Void})
    global initialized
    global finalized
    global libpython
    global pyprogramname
    global inspect
    global BuiltinFunctionType
    global TypeType
    global MethodType
    global MethodWrapperType
    global ufuncType
    global pynothing
    global PyNoneType
    if !initialized::Bool
        if finalized::Bool
            # From the Py_Finalize documentation:
            #    "Some extensions may not work properly if their
            #     initialization routine is called more than once; this
            #     can happen if an application calls Py_Initialize()
            #     and Py_Finalize() more than once."
            # For example, numpy and scipy seem to crash if this is done.
            error("Calling pyinitialize after pyfinalize is not supported")
        end
        libpython::Ptr{Void} = libpy == C_NULL ? ccall(:jl_load_dynamic_library, Ptr{Void}, (Ptr{Uint8},Cuint), C_NULL, 0) : libpy
        if 0 == ccall(pyfunc(:Py_IsInitialized), Cint, ())
            if !isempty(pyprogramname::ASCIIString)
                ccall(pyfunc(:Py_SetProgramName), Void, (Ptr{Uint8},), 
                      pyprogramname::ASCIIString)
            end
            ccall(pyfunc(:Py_InitializeEx), Void, (Cint,), 0)
        end
        initialized::Bool = true
        inspect::PyObject = pyimport("inspect")
        types = pyimport("types")
        BuiltinFunctionType::PyObject = types["BuiltinFunctionType"]
        TypeType::PyObject = types["TypeType"]
        MethodType::PyObject = types["MethodType"]
        MethodWrapperType::PyObject = pycall(TypeType::PyObject, PyObject, 
                                             PyObject(PyObject[])["__add__"])
        try
            ufuncType = pyimport("numpy")["ufunc"]
        catch
            ufuncType = PyObject(C_NULL) # NumPy not available
        end
        pynothing = pybuiltin("None")
        PyNoneType = types["NoneType"]
    end
    return
end

pyconfigvar(python::String, var::String) = chomp(readall(`$python -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('$var')"`))

function libpython_name(python::String)
    lib = pyconfigvar(python, "LDLIBRARY")
    @osx_only if lib[1] != '/'; lib = pyconfigvar(python, "PYTHONFRAMEWORKPREFIX")*"/"*lib end
    lib
end

# initialize the Python interpreter (no-op on subsequent calls)
function pyinitialize(python::String)
    global initialized
    global pyprogramname
    if !initialized::Bool
        libpy = dlopen(libpython_name(python),
                       RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        pyprogramname::ASCIIString = bytestring(python)
        pyinitialize(libpy)
    end
    return
end

pyinitialize() = pyinitialize("python") # default Python executable name
libpython_name() = libpython_name("python") # ditto

# end the Python interpreter and free associated memory
function pyfinalize()
    global initialized
    global finalized
    global libpython
    global inspect
    global BuiltinFunctionType
    global TypeType
    global MethodType
    global MethodWrapperType
    global ufuncType
    global pynothing
    global PyNoneType
    if initialized::Bool
        pydecref(ufuncType)
        npyfinalize()
        pydecref(PyNoneType)
        pydecref(pynothing)
        pydecref(BuiltinFunctionType::PyObject)
        pydecref(TypeType::PyObject)
        pydecref(MethodType::PyObject)
        pydecref(MethodWrapperType::PyObject)
        pydecref(inspect::PyObject)
        pygc_finalize()
        gc() # collect/decref any remaining PyObjects
        ccall(pyfunc(:Py_Finalize), Void, ())
        dlclose(libpython::Ptr{Void})
        libpython::Ptr{Void} = C_NULL
        initialized::Bool = false
        finalized::Bool = true
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

include("gc.jl")

# make a PyObject that embeds a reference to keep, to prevent Julia
# from garbage-collecting keep until o is finalized.
PyObject(o::PyPtr, keep::Any) = pyembed(PyObject(o), keep)

#########################################################################

include("callback.jl")

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
# For o::PyObject, make o["foo"] and o[:foo] equivalent to o.foo in Python,
# with the former returning an raw PyObject and the latter giving the PyAny
# conversion.

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

ref(o::PyObject, s::Symbol) = convert(PyAny, ref(o, string(s)))

function assign(o::PyObject, v, s::String)
    if (o.o == C_NULL)
        throw(ArgumentError("assign of NULL PyObject"))
    end
    if -1 == ccall(pyfunc(:PyObject_SetAttrString), Cint,
                   (PyPtr, Ptr{Uint8}, PyPtr), o, bytestring(s), PyObject(v))
        pyerr_clear()
        throw(KeyError(s))
    end
    o
end

assign(o::PyObject, v, s::Symbol) = assign(o, v, string(s))

#########################################################################
# Create anonymous composite w = pywrap(o) wrapping the object o
# and providing access to o's members (converted to PyAny) as w.member.

abstract PyWrapper

# still provide w["foo"] low-level access to unconverted members:
ref(w::PyWrapper, s) = ref(w.___jl_PyCall_PyObject___, s)

typesymbol(T::DataType) = T.name.name
typesymbol(T) = :Any # punt

function pywrap(o::PyObject)
    @pyinitialize
    members = convert(Vector{(String,PyObject)}, 
                      pycall(inspect["getmembers"], PyObject, o))
    tname = gensym("PyCall_PyWrapper")
    @eval begin
        $(Expr(:type, true, Expr(:<:, tname, :PyWrapper),
               Expr(:block, :(___jl_PyCall_PyObject___::PyObject),
                    map(m -> Expr(:(::), symbol(m[1]),
                                  typesymbol(pytype_query(m[2]))), 
                        members)...)))
        $(Expr(:call, tname, o,
               [ :(convert(PyAny, $(members[i][2])))
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
# Keyword arguments are just passed to pycall as @pykw kw1=val1 kw2=val2...
# in the last pycall argument.  The @pykw macro converts this into a special
# dictionary object for passing to pycall.
#
# Note that I don't use the Options package because a lot of the things
# like typechecking and defaults don't make sense the pycall context,
# so this can be a lot simpler.  All Python needs is a dictionary.

# In order for PyCall to differentiate between the keyword dictionary
# and an ordinary dictionary parameter, we wrap it in a special type.
type PyKW
    d::Dict{String,Any} # dictionary of (keyword => value) pairs.
end
PyObject(kw::PyKW) = isempty(kw.d) ? PyObject(C_NULL) : PyObject(kw.d)

# from a single :(x = y) expression make a single :(x => y) expression
function pykw1(ex::Expr)
    if ex.head == :(=) && length(ex.args) == 2 && typeof(ex.args[1]) == Symbol
        Expr(:(=>), string(ex.args[1]), ex.args[2])
    else
        throw(ArgumentError("invalid keyword expression $ex"))
    end
end

macro pykw(args...)
    :(PyKW($(Expr(:typed_dict, :(String=>Any), map(pykw1, args)...))))
end

#########################################################################

function pycall(o::PyObject, returntype::Union(Type,NTuple{Type}), args...)
    oargs = map(PyObject, args)
    nargs = length(args)
    if nargs > 0 && isa(args[end], PyKW)
        kw = PyObject(args[end])
        nargs -= 1
    else
        kw = PyObject(C_NULL)
    end
    arg = PyObject(@pycheckn ccall(pyfunc(:PyTuple_New), PyPtr, (Int,), 
                                   nargs))
    for i = 1:nargs
        @pycheckzi ccall(pyfunc(:PyTuple_SetItem), Cint, (PyPtr,Int,PyPtr),
                         arg, i-1, oargs[i])
        pyincref(oargs[i]) # PyTuple_SetItem steals the reference
    end
    ret = PyObject(@pycheckni ccall(pyfunc(:PyObject_Call), PyPtr,
                                    (PyPtr,PyPtr,PyPtr), o, arg, kw))
    jret = convert(returntype, ret)
    return jret
end

#########################################################################

end # module PyCall
