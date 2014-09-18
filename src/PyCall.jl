module PyCall

export pyinitialize, pyfinalize, pycall, pyimport, pybuiltin, PyObject,
       pysym, PyPtr, pyincref, pydecref, pyversion, PyArray, PyArray_Info,
       pyerr_check, pyerr_clear, pytype_query, PyAny, @pyimport, PyDict,
       pyisinstance, pywrap, pytypeof, pyeval, pyhassym, PyVector, pystring,
       pyraise, pytype_mapping, pygui, pygui_start, pygui_stop,
       pygui_stop_all, @pylab, set!, PyTextIO

import Base: size, ndims, similar, copy, getindex, setindex!, stride,
       convert, pointer, summary, convert, show, haskey, keys, values,
       eltype, get, delete!, empty!, length, isempty, start, done,
       next, filter!, hash, splice!, pop!, ==, isequal, help, push!,
       unshift!, shift!, append!, insert!, prepend!, writemime, mimewritable

# Python C API is not interrupt-save.  In principle, we should
# use sigatomic for every ccall to the Python library, but this
# should really be fixed in Julia (#2622).  However, we will 
# use the sigatomic_begin/end functions to protect pycall and
# similar long-running (or potentially long-running) code.
import Base: sigatomic_begin, sigatomic_end

#########################################################################

# Mirror of C PyObject struct (for non-debugging Python builds).  
# We won't actually access these fields directly; we'll use the Python
# C API for everything.  However, we need to define a unique Ptr type
# for PyObject*, and we might as well define the actual struct layout
# while we're at it.
immutable PyObject_struct
    ob_refcnt::Int
    ob_type::Ptr{Void}
end

typealias PyPtr Ptr{PyObject_struct} # type for PythonObject* in ccall

#########################################################################

# Global configuration variables.  Note that, since Julia does not allow us
# to attach type annotations to globals, we need to annotate these explicitly
# as initialized::Bool and libpython::Ptr{Void} when we use them.
initialized = false # whether Python is initialized
finalized = false # whether Python has been finalized
libpython = C_NULL # Python shared library (from dlopen)

pysym(func::Symbol) = dlsym(libpython::Ptr{Void}, func)
pysym_e(func::Symbol) = dlsym_e(libpython::Ptr{Void}, func)
pyhassym(func::Symbol) = pysym_e(func) != C_NULL

# call pysym_e on the arguments and return the first non-NULL result
function pysym_e(funcs...)
    for func in funcs
        p = pysym_e(func)
        if p != C_NULL
            return p
        end
    end
    return C_NULL
end

# Macro version of pysym to cache dlsym lookup (thanks to vtjnash)
macro pysym(func)
    z, zlocal = gensym(string(func)), gensym()
    eval(current_module(),:(global $z = C_NULL))
    z = esc(z)
    quote
        let $zlocal::Ptr{Void} = $z::Ptr{Void}
            if $zlocal == C_NULL
               global $z = $zlocal = dlsym(libpython::Ptr{Void}, $(esc(func)))
            end
            $zlocal
        end
    end
end

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
    PyObject() = PyObject(convert(PyPtr, C_NULL))
end

function pydecref(o::PyObject)
    if initialized::Bool # don't decref after pyfinalize!
        ccall((@pysym :Py_DecRef), Void, (PyPtr,), o.o)
    end
    o.o = convert(PyPtr, C_NULL)
    o
end

function pyincref(o::PyObject)
    ccall((@pysym :Py_IncRef), Void, (PyPtr,), o)
    o
end

# doing an incref *before* creating a PyObject may safer in the
# case of borrowed references, to ensure that no exception or interrupt
# induces a double decref.
function pyincref(o::PyPtr)
    ccall((@pysym :Py_IncRef), Void, (PyPtr,), o)
    PyObject(o)
end

pyisinstance(o::PyObject, t::PyObject) = 
  t.o != C_NULL && ccall((@pysym :PyObject_IsInstance), Cint, (PyPtr,PyPtr), o, t.o) == 1

pyisinstance(o::PyObject, t::Ptr{Void}) = 
  t != C_NULL && ccall((@pysym :PyObject_IsInstance), Cint, (PyPtr,PyPtr), o, t) == 1

pyquery(q::Ptr{Void}, o::PyObject) =
  ccall(q, Cint, (PyPtr,), o) == 1

pytypeof(o::PyObject) = o.o == C_NULL ? throw(ArgumentError("NULL PyObjects have no Python type")) : pycall(TypeType, PyObject, o)

# conversion to pass PyObject as ccall arguments:
convert(::Type{PyPtr}, po::PyObject) = po.o

# use constructor for generic conversions to PyObject
convert(::Type{PyObject}, o) = PyObject(o)
PyObject(o::PyObject) = o

#########################################################################

inspect = PyObject() # inspect module, needed for module introspection
builtin = PyObject() # __builtin__ module, needed for pybuiltin

# Python has zillions of types that a function be, in addition to the FunctionType
# in the C API.  We have to obtain these at runtime and cache them in globals
BuiltinFunctionType = PyObject()
TypeType = PyObject() # "type" function
MethodType = PyObject()
MethodWrapperType = PyObject()
# also WrapperDescriptorType = type(list.__add__) and
#      MethodDescriptorType = type(list.append) ... is it worth detecting these?

# special function type used in NumPy and SciPy (if available)
ufuncType = PyObject()

# cache Python None
pynothing = convert(PyPtr,C_NULL) # PyPtr, not PyObject, to prevent it from
                                  # being finalized prematurely on exit

# Python 2/3 compatibility: cache dlsym for renamed functions
pystring_fromstring = C_NULL
pystring_asstring = C_NULL
pystring_size = C_NULL
pystring_type = C_NULL
pyint_type = C_NULL
pyint_from_size_t = C_NULL
pyint_from_ssize_t = C_NULL
pyint_as_ssize_t = C_NULL

pyxrange = PyObject() # xrange type (or range in Python 3)

# cache ctypes.c_void_p type and function if available
c_void_p_Type = PyObject()
py_void_p = p::Ptr -> PyObject(uint(p))

# PyCObject_Check and PyCapsule_CheckExact are actually macros
# that check against PyCObject_Type and PyCapsule_Type global variables,
# which we cache if they are available:
PyCObject_Type = C_NULL
PyCapsule_Type = C_NULL

# Py_SetProgramName needs its argument to persist as long as Python does
pyprogramname = bytestring("")

# PyUnicode_* may actually be a #define for another symbol,
# so we cache the correct dlsym
PyUnicode_AsUTF8String = C_NULL
PyUnicode_DecodeUTF8 = C_NULL

# mpmath mpf/mpc types
mpmath = PyObject()
mpf = PyObject()
mpc = PyObject()

# whether to use unicode for strings by default, ala Python 3
pyunicode_literals = false

# traceback.format_tb function, for show(PyError)
format_traceback = PyObject()

# cache whether Python hash values are Clong (Python < 3.2) or Int (>= 3.2)
pyhashlong = false

# Return the Python version as a Julia VersionNumber
pyversion = v"0"

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
    global PyUnicode_AsUTF8String
    global PyUnicode_DecodeUTF8
    global pyunicode_literals
    global pynothing
    global pyxrange
    global pyhashlong
    global pystring_fromstring
    global pystring_asstring
    global pystring_size
    global pystring_type
    global pyint_type
    global pyint_from_size_t
    global pyint_from_ssize_t
    global pyint_as_ssize_t
    global pyversion
    global PyCObject_Type
    global PyCapsule_Type
    global c_void_p_Type
    global py_void_p
    global format_traceback
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
        already_inited = 0 != ccall((@pysym :Py_IsInitialized), Cint, ())
        if !already_inited
            if !isempty(pyprogramname)
                if pyversion.major < 3
                    ccall((@pysym :Py_SetProgramName), Void, (Ptr{Uint8},), 
                          pyprogramname::ASCIIString)
                else
                    ccall((@pysym :Py_SetProgramName), Void, (Ptr{Cwchar_t},), 
                          pyprogramname::Array{Cwchar_t})
                end
            end
            ccall((@pysym :Py_InitializeEx), Void, (Cint,), 0)
        end
        initialized::Bool = true
        inspect::PyObject = pyimport("inspect")
        types = pyimport("types")
        BuiltinFunctionType::PyObject = types["BuiltinFunctionType"]
        TypeType::PyObject = pybuiltin("type")
        MethodType::PyObject = types["MethodType"]
        MethodWrapperType::PyObject = pytypeof(PyObject(PyObject[])["__add__"])
        try
            ufuncType = pyimport("numpy")["ufunc"]
        catch
            ufuncType = PyObject() # NumPy not available
        end
        PyUnicode_AsUTF8String::Ptr{Void} =
          pysym_e(:PyUnicode_AsUTF8String,
                  :PyUnicodeUCS4_AsUTF8String,
                  :PyUnicodeUCS2_AsUTF8String)
        PyUnicode_DecodeUTF8::Ptr{Void} =
          pysym_e(:PyUnicode_DecodeUTF8,
                  :PyUnicodeUCS4_DecodeUTF8,
                  :PyUnicodeUCS2_DecodeUTF8)
        pynothing::PyPtr = convert(PyPtr, pysym(:_Py_NoneStruct))
        pyxrange::PyObject = pyincref(convert(PyPtr, pysym(:PyRange_Type)))
        if pyhassym(:PyString_FromString)
            pystring_fromstring::Ptr{Void} = pysym(:PyString_FromString)
            pystring_asstring::Ptr{Void} = pysym(:PyString_AsString)
            pystring_size::Ptr{Void} = pysym(:PyString_Size)
            pystring_type::Ptr{Void} = pysym(:PyString_Type)
        else
            pystring_fromstring::Ptr{Void} = pysym(:PyBytes_FromString)
            pystring_asstring::Ptr{Void} = pysym(:PyBytes_AsString)
            pystring_size::Ptr{Void} = pysym(:PyBytes_Size)
            pystring_type::Ptr{Void} = pysym(:PyBytes_Type)
        end
        if pyhassym(:PyInt_Type)
            pyint_type::Ptr{Void} = pysym(:PyInt_Type)
            pyint_from_size_t::Ptr{Void} = pysym(:PyInt_FromSize_t)
            pyint_from_ssize_t::Ptr{Void} = pysym(:PyInt_FromSsize_t)
            pyint_as_ssize_t::Ptr{Void} = pysym(:PyInt_AsSsize_t)
        else
            pyint_type::Ptr{Void} = pysym(:PyLong_Type)
            pyint_from_size_t::Ptr{Void} = pysym(:PyLong_FromSize_t)
            pyint_from_ssize_t::Ptr{Void} = pysym(:PyLong_FromSsize_t)
            pyint_as_ssize_t::Ptr{Void} = pysym(:PyLong_AsSsize_t)
        end
        PyCObject_Type::Ptr{Void} = pysym_e(:PyCObject_Type)
        PyCapsule_Type::Ptr{Void} = pysym_e(:PyCapsule_Type)
        try
            c_void_p_Type::PyObject = pyimport("ctypes")["c_void_p"]
            py_void_p::Function = p::Ptr -> pycall(c_void_p_Type::PyObject, PyObject, uint(p))
        catch # fallback to CObject
            pycobject_new = pysym(:PyCObject_FromVoidPtr)
            py_void_p::Function = p::Ptr -> PyObject(ccall(pycobject_new, PyPtr, (Ptr{Void}, Ptr{Void}), p, C_NULL))
        end
        pyversion::VersionNumber = 
          VersionNumber(convert((Int,Int,Int,String,Int), 
                                pyimport("sys")["version_info"])[1:3]...)
        pyhashlong::Bool = pyversion::VersionNumber < v"3.2"
        pyunicode_literals::Bool = pyversion::VersionNumber >= v"3.0"
        format_traceback::PyObject = pyimport("traceback")["format_tb"]
        pyexc_initialize()
        if !already_inited
            # some modules (e.g. IPython) expect sys.argv to be set
            if pyversion.major < 3
                argv_s = bytestring("")
                argv = convert(Ptr{Uint8}, argv_s)
                ccall(pysym(:PySys_SetArgvEx), Void, (Cint,Ptr{Ptr{Uint8}},Cint), 1, &argv, 0)
            else
                argv_s = Cwchar_t[0]
                argv   = convert(Ptr{Cwchar_t}, argv_s)
                ccall(pysym(:PySys_SetArgvEx), Void, (Cint, Ptr{Ptr{Cwchar_t}}, Cint), 1, &argv, 0)
            end
            
            # Some Python code checks sys.ps1 to see if it is running
            # interactively, and refuses to be interactive otherwise.
            # (e.g. Matplotlib: see PyPlot#79)
            if Base.is_interactive
                let sys = pyimport("sys")
                    if !haskey(sys, "ps1")
                        sys["ps1"] = ">>> "
                    end
                end
            end
        end
    end
    return
end

pyconfigvar(python::String, var::String) = chomp(readall(`$python -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('$var'))"`))
pysys(python::String, var::String) = chomp(readall(`$python -c "import sys; print(sys.$var)"`))
pyconfigvar(python, var, default) = let v = pyconfigvar(python, var)
    v == "None" ? default : v
end

const dlext = isdefined(Sys, :dlext) ? Sys.dlext : Sys.shlib_ext
const dlprefix = @windows? "" : "lib"

function dlopen_libpython(python::String)
    # it is ridiculous that it is this hard to find the name of libpython
    v = pyconfigvar(python,"VERSION","")
    libs = [ dlprefix*"python"*v*"."*dlext, dlprefix*"python."*dlext ]
    lib = pyconfigvar(python, "LIBRARY")
    lib != "None" && unshift!(libs, splitext(lib)[1]*dlext)
    lib = pyconfigvar(python, "LDLIBRARY")
    lib != "None" && unshift!(unshift!(libs, basename(lib)), lib)
    libs = unique(libs)

    libpaths = [pyconfigvar(python, "LIBDIR"),
                (@windows ? dirname(pysys(python, "executable")) : joinpath(dirname(dirname(pysys(python, "executable"))), "lib"))]
    @osx_only push!(libpaths, pyconfigvar(python, "PYTHONFRAMEWORKPREFIX"))

    # `prefix` and `exec_prefix` are the path prefixes where python should look for python only and compiled libraries, respectively.
    # These are also changed when run in a virtualenv.
    exec_prefix = pyconfigvar(python, "exec_prefix")
    # Since we only use `libpaths` to find the python dynamic library, we should only add `exec_prefix` to it.
    push!(libpaths, exec_prefix)
    if !haskey(ENV, "PYTHONHOME")
        # PYTHONHOME tells python where to look for both pure python
        # and binary modules.  When it is set, it replaces both
        # `prefix` and `exec_prefix` and we thus need to set it to
        # both in case they differ. This is also what the
        # documentation recommends.  However, they are documented
        # to always be the same on Windows, where it causes
        # problems if we try to include both.
        ENV["PYTHONHOME"] = @windows? exec_prefix : pyconfigvar(python, "prefix") * ":" * exec_prefix
        # Unfortunately, setting PYTHONHOME screws up Canopy's Python distro
        try
	    run(`$python -c "import site"` |> DevNull .> DevNull)
        catch
	    pop!(ENV, "PYTHONHOME")
        end
    end
    # TODO: look in python-config output? pyconfigvar("LDFLAGS")?
    for lib in libs
        for libpath in libpaths
     	    if isfile(joinpath(libpath, lib))
                try
                    return dlopen(joinpath(libpath, lib),
                                  RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
                end
            end
        end
    end

    # We do this *last* because the libpython in the system
    # library path might be the wrong one if multiple python
    # versions are installed (we prefer the one in LIBDIR):
    for lib in libs
        lib = splitext(lib)[1] 
        try
            return dlopen(lib, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        end
    end
    error("Couldn't find libpython; try pyinitialize(\"/path/to/libpython\")")
end

# Python 3.x uses wchar_t arrays for some string arguments
function wbytestring(s::String)
    if pyversion.major < 3 || sizeof(Cwchar_t) == 1 # ASCII (or UTF8)
        bytestring(s)
    else # UTF16 or UTF32, presumably
        if sizeof(Cwchar_t) == 4 # UTF32
            n = length(s)
            w = Array(Cwchar_t, n + 1)
            i = 0
            for c in s
                w[i += 1] = c
            end
        else # UTF16, presumably
            @assert sizeof(Cwchar_t) == 2
            if defined(UTF16String) # Julia 0.3
                s16 = utf16(s)
                w = Array(Cwchar_t, length(s.data)+1)
                copy!(w,1, s16.data,1, length(s.data))
            else
                n = length(s)
                w = Array(Cwchar_t, n + 1)
                i = 0
                for c in s
                    # punt on multi-byte encodings
                    c > 0xffff && error("unsupported char $c in \"$s\"")
                    w[i += 1] = c
                end
            end
        end
        w[end] = 0
        w
    end
end

# initialize the Python interpreter (no-op on subsequent calls)
function pyinitialize(python::String)
    global initialized
    if !initialized::Bool
        libpy = try
            lib = dlopen_libpython(python)
            global pyprogramname = wbytestring(pysys(python, "executable"))
            lib
        catch
            # perhaps we were passed library name and not executable?
            lib = dlopen(python, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
            global pyprogramname = wbytestring(python)
            lib
        end
        pyinitialize(libpy)
    end
    return
end

pyinitialize() = pyinitialize(get(ENV, "PYTHON", "python")) 
dlopen_libpython() = dlopen_libpython(get(ENV, "PYTHON", "python"))

# end the Python interpreter and free associated memory
function pyfinalize()
    global initialized
    global finalized
    global libpython
    global inspect
    global builtin
    global BuiltinFunctionType
    global TypeType
    global MethodType
    global MethodWrapperType
    global ufuncType
    global pyxrange
    global c_void_p_Type
    global format_traceback
    global mpmath
    global mpf
    global mpc
    if initialized::Bool
        pygui_stop_all()
        pydecref(mpc::PyObject)
        pydecref(mpf::PyObject)
        pydecref(mpmath::PyObject)
        pydecref(ufuncType::PyObject)
        npyfinalize()
        pydecref(format_traceback::PyObject)
        pydecref(c_void_p_Type::PyObject)
        pydecref(pyxrange::PyObject)
        pydecref(BuiltinFunctionType::PyObject)
        pydecref(TypeType::PyObject)
        pydecref(MethodType::PyObject)
        pydecref(MethodWrapperType::PyObject)
        pydecref(builtin::PyObject)
        pydecref(inspect::PyObject)
        pyexc_finalize()
        pycallback_finalize()
        pyio_finalize()
        pygc_finalize()
        gc() # collect/decref any remaining PyObjects
        ccall((@pysym :Py_Finalize), Void, ())
        dlclose(libpython::Ptr{Void})
        libpython::Ptr{Void} = C_NULL
        initialized::Bool = false
        finalized::Bool = true
    end
    return
end

#########################################################################

include("exception.jl")
include("gui.jl")

#########################################################################

include("gc.jl")

# make a PyObject that embeds a reference to keep, to prevent Julia
# from garbage-collecting keep until o is finalized.
PyObject(o::PyPtr, keep::Any) = pyembed(PyObject(o), keep)

#########################################################################

include("conversions.jl")

include("pytype.jl")

include("callback.jl")

include("io.jl")

#########################################################################
# Pretty-printing PyObject

function pystring(o::PyObject)
    if o.o == C_NULL
        return "NULL"
    else
        s = ccall((@pysym :PyObject_Repr), PyPtr, (PyPtr,), o)
        if (s == C_NULL)
            pyerr_clear()
            s = ccall((@pysym :PyObject_Str), PyPtr, (PyPtr,), o)
            if (s == C_NULL)
                pyerr_clear()
                return string(o.o)
            end
        end
        return convert(String, PyObject(s))
    end
end    

function show(io::IO, o::PyObject)
    print(io, "PyObject $(pystring(o))")
end

function help(o::PyObject)
    pycall(pybuiltin("help"), PyAny, o)
end

#########################################################################
# computing hashes of PyObjects

const pysalt = hash("PyCall.PyObject") # "salt" to mix in to PyObject hashes

# backwards compatibility with Julia 0.2
if VERSION < v"0.3-"
    hashsalt(x) = bitmix(pysalt, hash(x))
else
    hashsalt(x) = hash(x, pysalt)
end

function hash(o::PyObject)
    if o.o == C_NULL
        hashsalt(C_NULL)
    elseif is_pyjlwrap(o)
        # call native Julia hash directly on wrapped Julia objects,
        # since on 64-bit Windows the Python 2.x hash is only 32 bits
        hashsalt(unsafe_pyjlwrap_to_objref(o.o))
    else
        h = pyhashlong::Bool ? # changed to Py_hash_t in Python 3.2
               ccall((@pysym :PyObject_Hash), Clong, (PyPtr,), o) :
               ccall((@pysym :PyObject_Hash), Int, (PyPtr,), o)
        if h == -1 # error
            pyerr_clear()
            return hashsalt(o.o)
        end
        hashsalt(h)
    end
end

#########################################################################
# PyObject equality

const Py_EQ = convert(Cint, 2) # from Python's object.h

function ==(o1::PyObject, o2::PyObject)
    if o1.o == C_NULL || o2.o == C_NULL
        return o1.o == o2.o
    elseif is_pyjlwrap(o1)
        if is_pyjlwrap(o2)
            return unsafe_pyjlwrap_to_objref(o1.o) == 
                   unsafe_pyjlwrap_to_objref(o2.o)
        else
            return false
        end
    else
        val = ccall((@pysym :PyObject_RichCompareBool), Cint,
                    (PyPtr, PyPtr, Cint), o1, o2, Py_EQ)
        return val == -1 ? o1.o == o2.o : bool(val)
    end
end

isequal(o1::PyObject, o2::PyObject) = o1 == o2 # Julia 0.2 compatibility

#########################################################################
# For o::PyObject, make o["foo"] and o[:foo] equivalent to o.foo in Python,
# with the former returning an raw PyObject and the latter giving the PyAny
# conversion.

function getindex(o::PyObject, s::String)
    if (o.o == C_NULL)
        throw(ArgumentError("ref of NULL PyObject"))
    end
    p = ccall((@pysym :PyObject_GetAttrString), PyPtr,
              (PyPtr, Ptr{Uint8}), o, bytestring(s))
    if p == C_NULL
        pyerr_clear()
        throw(KeyError(s))
    end
    return PyObject(p)
end

getindex(o::PyObject, s::Symbol) = convert(PyAny, getindex(o, string(s)))

function setindex!(o::PyObject, v, s::String)
    if (o.o == C_NULL)
        throw(ArgumentError("assign of NULL PyObject"))
    end
    if -1 == ccall((@pysym :PyObject_SetAttrString), Cint,
                   (PyPtr, Ptr{Uint8}, PyPtr), o, bytestring(s), PyObject(v))
        pyerr_clear()
        throw(KeyError(s))
    end
    o
end

setindex!(o::PyObject, v, s::Symbol) = setindex!(o, v, string(s))

function haskey(o::PyObject, s::String)
    if (o.o == C_NULL)
        throw(ArgumentError("haskey of NULL PyObject"))
    end
    return 1 == ccall((@pysym :PyObject_HasAttrString), Cint,
                      (PyPtr, Ptr{Uint8}), o, bytestring(s))
end

haskey(o::PyObject, s::Symbol) = haskey(o, string(s))

#########################################################################
# keys(o) should return an iterator over the keys (members) of o

type PyObjectMembers
    members::PyVector{(Symbol,PyObject)}
end

function show(io::IO, m::PyObjectMembers)
    print(io, "PyObjectMembers iterator (", length(m), " members)")
end

keys(o::PyObject) = PyObjectMembers(pycall(inspect["getmembers"],
                                           PyVector{(Symbol,PyObject)}, o))

start(m::PyObjectMembers) = 1
done(m::PyObjectMembers, i::Int) = i > length(m.members)
next(m::PyObjectMembers, i::Int) = (m.members[i][1], i+1)
length(m::PyObjectMembers) = length(m.members)

#########################################################################
# Create anonymous composite w = pywrap(o) wrapping the object o
# and providing access to o's members (converted to PyAny) as w.member.

# we skip wrapping Julia reserved words (which cannot be type members)
const reserved = Set{ASCIIString}()
for w in ("while", "if", "for", "try", "return", "break", 
          "continue", "function", "macro", "quote", "let", "local",
          "global", "const", "abstract", "typealias", "type",
          "bitstype", "immutable", "ccall", "do", "module",
          "baremodule", "using", "import", "export", "importall",
          "pymember", "false", "true")
    push!(reserved, w) # construct Set this way for compat with Julia 0.2/0.3
end

function pywrap(o::PyObject, mname::Symbol=:__anon__)
    @pyinitialize
    members = convert(Vector{(String,PyObject)}, 
                      pycall(inspect["getmembers"], PyObject, o))
    filter!(m -> !(m[1] in reserved), members)
    m = Module(mname)
    consts = [Expr(:const, Expr(:(=), symbol(x[1]), convert(PyAny, x[2]))) for x in members]
    exports = try
                  convert(Vector{Symbol}, o["__all__"])
              catch
                  [symbol(x[1]) for x in filter(x -> x[1][1] != '_', members)]
              end
    eval(m, Expr(:toplevel, consts..., :(pymember(s) = getindex($(o), s)),
                 Expr(:export, exports...)))
    m
end

#########################################################################

pyimport(name::String) =
    PyObject(@pycheckn ccall((@pysym :PyImport_ImportModule), PyPtr,
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

# separate this function in order to make it easier to write more
# pyimport-like functions
function pyimport_name(name, optional_varname)
    len = length(optional_varname)
    if len > 0 && (len != 2 || optional_varname[1] != :as)
        throw(ArgumentError("usage @pyimport module [as name]"))
    elseif len == 2
        optional_varname[2]
    elseif typeof(name) == Symbol
        name
    else
        mname = modulename(name)
        throw(ArgumentError("$mname is not a valid module variable name, use @pyimport $mname as <name>"))
    end
end

macro pyimport(name, optional_varname...)
    mname = modulename(name)
    Name = pyimport_name(name, optional_varname)
    quote
        if !isdefined($(Expr(:quote, Name)))
            const $(esc(Name)) = pywrap(pyimport($mname))
        elseif !isa($(esc(Name)), Module)
            error("@pyimport: ", $(Expr(:quote, Name)), " already defined")
        end
        nothing
    end
end

#########################################################################

# look up a global builtin
function pybuiltin(name)
    global builtin
    if (builtin::PyObject).o == C_NULL
        builtin::PyObject = try
            pyimport("__builtin__")
        catch
            pyimport("builtins") # renamed in Python 3
        end
    end
    (builtin::PyObject)[name]
end

#########################################################################

typealias TypeTuple Union(Type,NTuple{Type})

function pycall(o::PyObject, returntype::TypeTuple, args...; kwargs...)
    oargs = map(PyObject, args)
    nargs = length(args)
    sigatomic_begin()
    try
        arg = PyObject(@pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,), 
                                       nargs))
        for i = 1:nargs
            @pycheckzi ccall((@pysym :PyTuple_SetItem), Cint,
                             (PyPtr,Int,PyPtr), arg, i-1, oargs[i])
            pyincref(oargs[i]) # PyTuple_SetItem steals the reference
        end
        if isempty(kwargs)
            ret = PyObject(@pycheckni ccall((@pysym :PyObject_Call), PyPtr,
                                          (PyPtr,PyPtr,PyPtr), o, arg, C_NULL))
        else
            kw = PyObject((String=>Any)[string(k) => v for (k, v) in kwargs])
            ret = PyObject(@pycheckni ccall((@pysym :PyObject_Call), PyPtr,
                                            (PyPtr,PyPtr,PyPtr), o, arg, kw))
        end
        jret = convert(returntype, ret)
        return jret
    finally
        sigatomic_end()
    end
end

#########################################################################
# Once Julia lets us overload ".", we will use [] to access items, but
# for now we can define "get"

function get(o::PyObject, returntype::TypeTuple, k, default) 
    r = ccall((@pysym :PyObject_GetItem), PyPtr, (PyPtr,PyPtr), o,PyObject(k))
    if r == C_NULL
        pyerr_clear()
        default
    else
        convert(returntype, PyObject(r))
    end
end

get(o::PyObject, returntype::TypeTuple, k) = 
    convert(returntype, PyObject(@pycheckni ccall((@pysym :PyObject_GetItem), 
                                 PyPtr, (PyPtr,PyPtr), o, PyObject(k))))

get(o::PyObject, k, default) = get(o, PyAny, k, default)
get(o::PyObject, k) = get(o, PyAny, k)

function delete!(o::PyObject, k)
    e = ccall((@pysym :PyObject_DelItem), Cint, (PyPtr, PyPtr), o, PyObject(k))
    if e == -1
        pyerr_clear() # delete! ignores errors in Julia
    end
    return o
end

function set!(o::PyObject, k, v)
    @pycheckzi ccall((@pysym :PyObject_SetItem), Cint, (PyPtr, PyPtr, PyPtr),
                     o, PyObject(k), PyObject(v))
    v
end

#########################################################################
# support IPython _repr_foo functions for writemime of PyObjects

for (mime, method) in ((MIME"text/html", "_repr_html_"),
                       (MIME"image/jpeg", "_repr_jpeg_"),
                       (MIME"image/png", "_repr_png_"),
                       (MIME"image/svg+xml", "_repr_svg_"),
                       (MIME"text/latex", "_repr_latex_"))
    T = istext(mime()) ? String : Vector{Uint8}
    @eval begin
        function writemime(io::IO, mime::$mime, o::PyObject)
            if o.o != C_NULL && haskey(o, $method)
                r = pycall(o[$method], PyObject)
                r.o != pynothing::PyPtr && return write(io, convert($T, r))
            end
            throw(MethodError(writemime, (io, mime, o)))
        end
        mimewritable(::$mime, o::PyObject) =
            o.o != C_NULL && haskey(o, $method) &&
            pycall(o[$method], PyObject).o != pynothing::PyPtr
    end
end

#########################################################################

const Py_eval_input = 258 # from Python.h
const pyeval_fname = bytestring("PyCall.jl") # filename for pyeval

# evaluate a python string, returning PyObject, given a dictionary
# (string/symbol => value) of local variables to use in the expression
function pyeval_(s::String, locals::PyDict) 
    sb = bytestring(s) # use temp var to prevent gc before we are done with o
    sigatomic_begin()
    try
        o = PyObject(@pycheckn ccall((@pysym :Py_CompileString), PyPtr,
                                     (Ptr{Uint8}, Ptr{Uint8}, Cint),
                                     sb, pyeval_fname, Py_eval_input))
        main = @pycheckni ccall((@pysym :PyImport_AddModule),
                                PyPtr, (Ptr{Uint8},),
                                bytestring("__main__"))
        maindict = @pycheckni ccall((@pysym :PyModule_GetDict), PyPtr,
                                    (PyPtr,), main)
        return PyObject(@pycheckni ccall((@pysym :PyEval_EvalCode),
                                         PyPtr, (PyPtr, PyPtr, PyPtr),
                                         o, maindict, locals))
    finally
        sigatomic_end()
    end
end

function pyeval(s::String, returntype::TypeTuple=PyAny; kwargs...)
    locals = PyDict{String,PyObject}()
    for (k, v) in kwargs
        locals[string(k)] = v
    end
    return convert(returntype, pyeval_(s, locals))
end

#########################################################################

end # module PyCall
