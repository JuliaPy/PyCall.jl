__precompile__()

module PyCall

export pycall, pyimport, pybuiltin, PyObject, PyReverseDims,
       PyPtr, pyincref, pydecref, pyversion, PyArray, PyArray_Info,
       pyerr_check, pyerr_clear, pytype_query, PyAny, @pyimport, PyDict,
       pyisinstance, pywrap, pytypeof, pyeval, PyVector, pystring,
       pyraise, pytype_mapping, pygui, pygui_start, pygui_stop,
       pygui_stop_all, @pylab, set!, PyTextIO, @pysym, PyNULL, @pydef,
       pyimport_conda, @py_str

import Base: size, ndims, similar, copy, getindex, setindex!, stride,
       convert, pointer, summary, convert, show, haskey, keys, values,
       eltype, get, delete!, empty!, length, isempty, start, done,
       next, filter!, hash, splice!, pop!, ==, isequal, push!,
       unshift!, shift!, append!, insert!, prepend!, mimewritable

# Python C API is not interrupt-save.  In principle, we should
# use sigatomic for every ccall to the Python library, but this
# should really be fixed in Julia (#2622).  However, we will
# use the sigatomic_begin/end functions to protect pycall and
# similar long-running (or potentially long-running) code.
import Base: sigatomic_begin, sigatomic_end

## Compatibility import for v0.4, v0.5
using Compat
import Conda
import Compat.String
@compat import Base.show
import Base.unsafe_convert
import MacroTools   # because of issue #270

if isdefined(Base, :Iterators)
    import Base.Iterators: filter
end

#########################################################################

include(joinpath(dirname(@__FILE__), "..", "deps","depsutils.jl"))
include("startup.jl")

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

const PyPtr = Ptr{PyObject_struct} # type for PythonObject* in ccall

const PyPtr_NULL = PyPtr(C_NULL)

#########################################################################
# Wrapper around Python's C PyObject* type, with hooks to Python reference
# counting and conversion routines to/from C and Julia types.
"""
    PyObject(juliavar)

This converts a julia variable to a PyObject, which is a reference to a Python object.
You can convert back to native julia types using `convert(T, o::PyObject)`, or using `PyAny(o)`.

Given `o::PyObject`, `o[:attribute]` is equivalent to `o.attribute` in Python, with automatic type conversion.

Given `o::PyObject`, `get(o, key)` is equivalent to `o[key]` in Python, with automatic type conversion.
"""
type PyObject
    o::PyPtr # the actual PyObject*
    function PyObject(o::PyPtr)
        po = new(o)
        finalizer(po, pydecref)
        return po
    end
end

"""
    PyNULL()

Return a `PyObject` that has a `NULL` underlying pointer, i.e. it doesn't
actually refer to any Python object.

This is useful for initializing `PyObject` global variables and array elements
before an actual Python object is available.   For example, you might do `const
myglobal = PyNULL()` and later on (e.g. in a module `__init__` function),
reassign `myglobal` to point to an actual object with `copy!(myglobal,
someobject)`.   This procedure will properly handle Python's reference counting
(so that the Python object will not be freed until you are done with
`myglobal`).
"""
PyNULL() = PyObject(PyPtr_NULL)

function pydecref(o::PyObject)
    ccall(@pysym(:Py_DecRef), Void, (PyPtr,), o.o)
    o.o = PyPtr_NULL
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

"""
"Steal" a reference from a PyObject: return the raw PyPtr, while
setting the corresponding `o.o` field to `NULL` so that no decref
will be performed when `o` is garbage collected.  (This means that
you can no longer use `o`.)  Used for passing objects to Python.
"""
function pystealref!(o::PyObject)
    optr = o.o
    o.o = PyPtr_NULL # don't decref when o is gc'ed
    return optr
end

function Base.copy!(dest::PyObject, src::PyObject)
    pydecref(dest)
    dest.o = src.o
    return pyincref(dest)
end

pyisinstance(o::PyObject, t::PyObject) =
  t.o != C_NULL && ccall((@pysym :PyObject_IsInstance), Cint, (PyPtr,PyPtr), o, t.o) == 1

pyisinstance(o::PyObject, t::Union{Ptr{Void},PyPtr}) =
  t != C_NULL && ccall((@pysym :PyObject_IsInstance), Cint, (PyPtr,PyPtr), o, t) == 1

pyquery(q::Ptr{Void}, o::PyObject) =
  ccall(q, Cint, (PyPtr,), o) == 1

pytypeof(o::PyObject) = o.o == C_NULL ? throw(ArgumentError("NULL PyObjects have no Python type")) : pycall(TypeType, PyObject, o)

# conversion to pass PyObject as ccall arguments:
unsafe_convert(::Type{PyPtr}, po::PyObject) = po.o

# use constructor for generic conversions to PyObject
convert(::Type{PyObject}, o) = PyObject(o)
convert(::Type{PyObject}, o::PyObject) = o
PyObject(o::PyObject) = o

#########################################################################

include("pyinit.jl")
include("exception.jl")
include("gui.jl")

#########################################################################

include("gc.jl")

# make a PyObject that embeds a reference to keep, to prevent Julia
# from garbage-collecting keep until o is finalized.
PyObject(o::PyPtr, keep::Any) = pyembed(PyObject(o), keep)

#########################################################################

@compat TypeTuple{N} = Union{Type,NTuple{N, Type}}
include("pybuffer.jl")
include("conversions.jl")
include("pytype.jl")
include("pyclass.jl")
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
        return convert(AbstractString, PyObject(s))
    end
end

function show(io::IO, o::PyObject)
    print(io, "PyObject $(pystring(o))")
end

function Base.Docs.doc(o::PyObject)
    Base.Docs.Text(haskey(o, "__doc__") ?
                   convert(AbstractString, o["__doc__"]) :
                   "Python object (no docstring found)")
end

#########################################################################
# computing hashes of PyObjects

const pysalt = hash("PyCall.PyObject") # "salt" to mix in to PyObject hashes
hashsalt(x) = hash(x, pysalt)

function hash(o::PyObject)
    if o.o == C_NULL
        hashsalt(C_NULL)
    elseif is_pyjlwrap(o)
        # call native Julia hash directly on wrapped Julia objects,
        # since on 64-bit Windows the Python 2.x hash is only 32 bits
        hashsalt(unsafe_pyjlwrap_to_objref(o.o))
    else
        h = ccall((@pysym :PyObject_Hash), Py_hash_t, (PyPtr,), o)
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
        return val == -1 ? o1.o == o2.o : Bool(val)
    end
end

isequal(o1::PyObject, o2::PyObject) = o1 == o2 # Julia 0.2 compatibility

#########################################################################
# For o::PyObject, make o["foo"] and o[:foo] equivalent to o.foo in Python,
# with the former returning an raw PyObject and the latter giving the PyAny
# conversion.

function getindex(o::PyObject, s::AbstractString)
    if (o.o == C_NULL)
        throw(ArgumentError("ref of NULL PyObject"))
    end
    p = ccall((@pysym :PyObject_GetAttrString), PyPtr, (PyPtr, Cstring), o, s)
    if p == C_NULL
        pyerr_clear()
        throw(KeyError(s))
    end
    return PyObject(p)
end

getindex(o::PyObject, s::Symbol) = convert(PyAny, getindex(o, string(s)))

function setindex!(o::PyObject, v, s::Union{Symbol,AbstractString})
    if (o.o == C_NULL)
        throw(ArgumentError("assign of NULL PyObject"))
    end
    if -1 == ccall((@pysym :PyObject_SetAttrString), Cint,
                   (PyPtr, Cstring, PyPtr), o, s, PyObject(v))
        pyerr_clear()
        throw(KeyError(s))
    end
    o
end

function haskey(o::PyObject, s::Union{Symbol,AbstractString})
    if (o.o == C_NULL)
        throw(ArgumentError("haskey of NULL PyObject"))
    end
    return 1 == ccall((@pysym :PyObject_HasAttrString), Cint,
                      (PyPtr, Cstring), o, s)
end

#########################################################################

keys(o::PyObject) = Symbol[m[1] for m in pycall(inspect["getmembers"],
                                PyVector{Tuple{Symbol,PyObject}}, o)]

#########################################################################
# Create anonymous composite w = pywrap(o) wrapping the object o
# and providing access to o's members (converted to PyAny) as w.member.

# we skip wrapping Julia reserved words (which cannot be type members)
const reserved = Set{String}(["while", "if", "for", "try", "return", "break", "continue", "function", "macro", "quote", "let", "local", "global", "const", "abstract", "typealias", "type", "bitstype", "immutable", "ccall", "do", "module", "baremodule", "using", "import", "export", "importall", "pymember", "false", "true", "Tuple"])

"""
    pywrap(o::PyObject)

This returns a wrapper `w` that is an anonymous module which provides (read) access to converted versions of o's members as w.member.

For example, `@pyimport module as name` is equivalent to `const name = pywrap(pyimport("module"))`

If the Python module contains identifiers that are reserved words in Julia (e.g. function), they cannot be accessed as `w.member`; one must instead use `w.pymember(:member)` (for the PyAny conversion) or w.pymember("member") (for the raw PyObject).
"""
function pywrap(o::PyObject, mname::Symbol=:__anon__)
    members = convert(Vector{Tuple{AbstractString,PyObject}},
                      pycall(inspect["getmembers"], PyObject, o))
    filter!(m -> !(m[1] in reserved), members)
    m = Module(mname, false)
    consts = [Expr(:const, Expr(:(=), Symbol(x[1]), convert(PyAny, x[2]))) for x in members]
    exports = try
                  convert(Vector{Symbol}, o["__all__"])
              catch
                  [Symbol(x[1]) for x in filter(x -> x[1][1] != '_', members)]
              end
    eval(m, Expr(:toplevel, consts..., :(pymember(s) = $(getindex)($(o), s)),
                 Expr(:export, exports...)))
    m
end

#########################################################################

@static if is_windows()
    # Many python extensions are linked against a very specific version of the
    # MSVC runtime library. To load this library, libpython declares an
    # appropriate manifest, but unfortunately most extensions do not.
    # Libpython itself does extend its activation context to any extensions it
    # loads, but some python libraries (e.g. libzmq), load such extensions
    # through ctypes, which does not have this functionality. Work around
    # this by manually activating the activation context before any call to
    # PyImport_ImportModule, since extensions are most likely to be loaded
    # during import.

    immutable ACTIVATION_CONTEXT_BASIC_INFORMATION
        handle::Ptr{Void}
        dwFlags::UInt32
    end
    const ActivationContextBasicInformation = 1
    const QUERY_ACTCTX_FLAG_ACTCTX_IS_ADDRESS = 0x10
    const PyActCtx = Ref{Ptr{Void}}(C_NULL)

    function ComputePyActCtx()
        if PyActCtx[] == C_NULL
            some_address_in_libpython = @pyglobal(:PyImport_ImportModule)
            ActCtxBasicInfo = Ref{ACTIVATION_CONTEXT_BASIC_INFORMATION}()
            succeeded = ccall(:QueryActCtxW,stdcall,Bool,
                (UInt32,Ptr{Void},Ptr{Void},Culong,Ptr{Void},Csize_t,Ptr{Csize_t}),
                QUERY_ACTCTX_FLAG_ACTCTX_IS_ADDRESS, some_address_in_libpython,
                C_NULL, ActivationContextBasicInformation, ActCtxBasicInfo,
                sizeof(ActCtxBasicInfo), C_NULL)
            @assert succeeded
            PyActCtx[] = ActCtxBasicInfo[].handle
        end
        PyActCtx[]
    end

    function ActivatePyActCtx()
        cookie = Ref{Ptr{Void}}()
        succeeded = ccall(:ActivateActCtx,stdcall, Bool,
                          (Ptr{Void}, Ptr{Ptr{Void}}),
                          ComputePyActCtx(), cookie)
        @assert succeeded
        cookie[]
    end

    function DeactivatePyActCtx(cookie)
        succeeded = ccall(:DeactivateActCtx, stdcall, Bool,
                         (UInt32, Ptr{Void}), 0, cookie)
        @assert succeeded
    end
else
    ActivatePyActCtx() = nothing
    DeactivatePyActCtx(cookie) = nothing
end

"""
    pyimport(s::AbstractString)

Import the Python module `s` (a string or symbol) and return a pointer to it (a `PyObject`). Functions or other symbols in the module may then be looked up by s[name] where name is a string (for the raw PyObject) or symbol (for automatic type-conversion). Unlike the @pyimport macro, this does not define a Julia module and members cannot be accessed with `s.name`
"""
function pyimport(name::AbstractString)
    cookie = ActivatePyActCtx()
    obj = PyObject(@pycheckn ccall((@pysym :PyImport_ImportModule), PyPtr,
                             (Cstring,), name))
    DeactivatePyActCtx(cookie)
    obj
end
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

"""
    anaconda_conda()

Return the path of the `conda` program if PyCall is configured to use
an Anaconda install (but *not* the Conda.jl Python), and the empty
string otherwise.
"""
function anaconda_conda()
    # Anaconda Python seems to always include "Anaconda" in the version string.
    if conda || !contains(unsafe_string(ccall(@pysym(:Py_GetVersion), Ptr{UInt8}, ())), "conda")
        return ""
    end
    aconda = joinpath(dirname(pyprogramname), "conda")
    return isfile(aconda) ? aconda : ""
end

"""
    pyimport_conda(modulename, condapkg, [channel])

Returns the result of `pyimport(modulename)` if possible.   If the module
is not found, and PyCall is configured to use the Conda Python distro
(via the Julia Conda package), then automatically install `condapkg`
via `Conda.add(condapkg)` and then re-try the `pyimport`.   Other
Anaconda-based Python installations are also supported as long as
their `conda` program is functioning.

If PyCall is not using Conda and the `pyimport` fails, throws
an exception with an error message telling the user how to configure
PyCall to use Conda for automated installation of the module.

The third argument, `channel` is an optional Anaconda "channel" to use
for installing the package; this is useful for packages that are not
included in the default Anaconda package listing.
"""
function pyimport_conda(modulename::AbstractString, condapkg::AbstractString,
                        channel::AbstractString="")
    try
        pyimport(modulename)
    catch e
        if conda
            info("Installing $modulename via the Conda $condapkg package...")
            isempty(channel) || Conda.add_channel(channel)
            Conda.add(condapkg)
            pyimport(modulename)
        else
            aconda = anaconda_conda()
            if !isempty(aconda)
                try
                    info("Installing $modulename via Anaconda's $aconda...")
                    isempty(channel) || run(`$aconda config --add channels $channel --force`)
                    run(`$aconda install -y $condapkg`)
                catch e2
                    error("""
                    Detected Anaconda Python, but \"$aconda install -y $condapkg\"  failed.

                    You should either install $condapkg manually or fix your Anaconda installation so that $aconda works.   Alternatively, you can reconfigure PyCall to use its own Miniconda installation via the Conda.jl package, by running:
                        ENV["PYTHON"]=""; Pkg.build("PyCall")
                    after re-launching Julia.

                    The exception was: $e2
                    """)
                end
                return pyimport(modulename)
            end

            error("""
                Failed to import required Python module $modulename.

                For automated $modulename installation, try configuring PyCall to use the Conda.jl package's Python "Miniconda" distribution within Julia. Relaunch Julia and run:
                    ENV["PYTHON"]=""
                    Pkg.build("PyCall")
                before trying again.

                The pyimport exception was: $e
                """)
        end
    end
end

#########################################################################

# look up a global builtin
"""
    pybuiltin(s::AbstractString)

Look up a string or symbol `s` among the global Python builtins. If `s` is a string it returns a PyObject, while if `s` is a symbol it returns the builtin converted to `PyAny`.
"""
function pybuiltin(name)
    builtin[name]
end

#########################################################################

"""
Low-level version of `pycall(o, ...)` that always returns `PyObject`.
"""
function _pycall(o::Union{PyObject,PyPtr}, args...; kwargs...)
    oargs = map(PyObject, args)
    nargs = length(args)
    sigatomic_begin()
    try
        arg = PyObject(@pycheckn ccall((@pysym :PyTuple_New), PyPtr, (Int,),
                                       nargs))
        for i = 1:nargs
            @pycheckz ccall((@pysym :PyTuple_SetItem), Cint,
                             (PyPtr,Int,PyPtr), arg, i-1, oargs[i])
            pyincref(oargs[i]) # PyTuple_SetItem steals the reference
        end
        if isempty(kwargs)
            ret = PyObject(@pycheckn ccall((@pysym :PyObject_Call), PyPtr,
                                          (PyPtr,PyPtr,PyPtr), o, arg, C_NULL))
        else
            #kw = PyObject((AbstractString=>Any)[string(k) => v for (k, v) in kwargs])
            kw = PyObject(Dict{AbstractString, Any}([Pair(string(k), v) for (k, v) in kwargs]))
            ret = PyObject(@pycheckn ccall((@pysym :PyObject_Call), PyPtr,
                                            (PyPtr,PyPtr,PyPtr), o, arg, kw))
        end
        return ret::PyObject
    finally
        sigatomic_end()
    end
end

"""
    pycall(o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...)

Call the given Python function (typically looked up from a module) with the given args... (of standard Julia types which are converted automatically to the corresponding Python types if possible), converting the return value to returntype (use a returntype of PyObject to return the unconverted Python object reference, or of PyAny to request an automated conversion)
"""
pycall(o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...) =
    return convert(returntype, _pycall(o, args...; kwargs...))::returntype

pycall(o::Union{PyObject,PyPtr}, ::Type{PyAny}, args...; kwargs...) =
    return convert(PyAny, _pycall(o, args...; kwargs...))

@compat (o::PyObject)(args...; kws...) = pycall(o, PyAny, args...; kws...)
@compat (::Type{PyAny})(o::PyObject) = convert(PyAny, o)


#########################################################################
# Once Julia lets us overload ".", we will use [] to access items, but
# for now we can define "get".

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
    convert(returntype, PyObject(@pycheckn ccall((@pysym :PyObject_GetItem),
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
    @pycheckz ccall((@pysym :PyObject_SetItem), Cint, (PyPtr, PyPtr, PyPtr),
                     o, PyObject(k), PyObject(v))
    v
end

#########################################################################
# Support [] for integer keys, and other duck-typed sequence/list operations,
# as those don't conflict with symbols/strings used for attributes.

# Index conversion: Python is zero-based.  It also has -1 based
# backwards indexing, but we don't support this, in favor of the
# Julian syntax o[end-1] etc.
function ind2py(i::Integer)
    i <= 0 && throw(BoundsError())
    return i-1
end

_getindex(o::PyObject, i::Integer, T) = convert(T, PyObject(@pycheckn ccall((@pysym :PySequence_GetItem), PyPtr, (PyPtr, Int), o, ind2py(i))))
getindex(o::PyObject, i::Integer) = _getindex(o, i, PyAny)
function setindex!(o::PyObject, v, i::Integer)
    @pycheckz ccall((@pysym :PySequence_SetItem), Cint, (PyPtr, Int, PyPtr), o, ind2py(i), PyObject(v))
    v
end
getindex(o::PyObject, i1::Integer, i2::Integer) = get(o, (ind2py(i1),ind2py(i2)))
setindex!(o::PyObject, v, i1::Integer, i2::Integer) = set!(o, (ind2py(i1),ind2py(i2)), v)
getindex(o::PyObject, I::Integer...) = get(o, map(ind2py, I))
setindex!(o::PyObject, v, I::Integer...) = set!(o, map(ind2py, I), v)
Base.endof(o::PyObject) = length(o)
length(o::PyObject) = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)

function splice!(a::PyObject, i::Integer)
    v = a[i]
    @pycheckz ccall((@pysym :PySequence_DelItem), Cint, (PyPtr, Int), a, i-1)
    v
end

pop!(a::PyObject) = splice!(a, length(a))
shift!(a::PyObject) = splice!(a, 1)

function empty!(a::PyObject)
    for i in length(a):-1:1
        @pycheckz ccall((@pysym :PySequence_DelItem), Cint, (PyPtr, Int), a, i-1)
    end
    a
end

# The following operations only work for the list type and subtypes thereof:
function push!(a::PyObject, item)
    @pycheckz ccall((@pysym :PyList_Append), Cint, (PyPtr, PyPtr),
                     a, PyObject(item))
    a
end

function insert!(a::PyObject, i::Integer, item)
    @pycheckz ccall((@pysym :PyList_Insert), Cint, (PyPtr, Int, PyPtr),
                     a, ind2py(i), PyObject(item))
    a
end

unshift!(a::PyObject, item) = insert!(a, 1, item)

function prepend!(a::PyObject, items)
    for (i,x) in enumerate(items)
        insert!(a, i, x)
    end
    a
end

function append!(a::PyObject, items)
    for item in items
        push!(a, item)
    end
    return a
end

#########################################################################
# support IPython _repr_foo functions for MIME output of PyObjects

for (mime, method) in ((MIME"text/html", "_repr_html_"),
                       (MIME"text/markdown", "_repr_markdown_"),
                       (MIME"text/json", "_repr_json_"),
                       (MIME"application/javascript", "_repr_javascript_"),
                       (MIME"application/pdf", "_repr_pdf_"),
                       (MIME"image/jpeg", "_repr_jpeg_"),
                       (MIME"image/png", "_repr_png_"),
                       (MIME"image/svg+xml", "_repr_svg_"),
                       (MIME"text/latex", "_repr_latex_"))
    T = istextmime(mime()) ? AbstractString : Vector{UInt8}
    @eval begin
        @compat function show(io::IO, mime::$mime, o::PyObject)
            if o.o != C_NULL && haskey(o, $method)
                r = pycall(o[$method], PyObject)
                r.o != pynothing[] && return write(io, convert($T, r))
            end
            throw(MethodError($(VERSION < v"0.5.0-dev+4340") ? writemime : show,
                              (io, mime, o)))
        end
        mimewritable(::$mime, o::PyObject) =
            o.o != C_NULL && haskey(o, $method) && let meth = o[$method]
                meth.o != pynothing[] &&
                pycall(meth, PyObject).o != pynothing[]
            end
    end
end

#########################################################################
# Expose Python docstrings to the Julia doc system

if isdefined(Docs, :getdoc)  # Introduced in Julia .6
    Docs.getdoc(o::PyObject) = Text(String(o["__doc__"]))
end


#########################################################################

include("pyeval.jl")

#########################################################################
# Precompilation: just an optimization to speed up initialization.
# Here, we precompile functions that are passed to cfunction by __init__,
# for the reasons described in JuliaLang/julia#12256.

precompile(jl_Function_call, (PyPtr,PyPtr,PyPtr))
precompile(pyjlwrap_dealloc, (PyPtr,))
precompile(pyjlwrap_repr, (PyPtr,))
precompile(pyjlwrap_hash, (PyPtr,))
precompile(pyjlwrap_hash32, (PyPtr,))

# TODO: precompilation of the io.jl functions

end # module PyCall
