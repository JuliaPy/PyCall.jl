#########################################################################
# PyAny and introspection
#
# PyAny is a "fake" type used to convert a PyObject to any native Julia
# type, if possible, via introspection of the Python type.

#########################################################################

abstract type PyAny end

# PyAny acts like Any for conversions, except for converting PyObject (below)
PyAny(x) = x

#########################################################################
# for automatic conversions, I pass Vector{PyAny}, NTuple{N, PyAny}, etc.,
# but since PyAny is an abstract type I need to convert this to Any
# before actually creating the Julia object
function pyany_toany(T::Type)
    T === Vararg{PyAny} ? Vararg{Any} : T
end
pyany_toany(::Type{PyAny}) = Any
pyany_toany(t::Type{T}) where {T<:Tuple} = Tuple{map(pyany_toany, t.types)...}

#########################################################################
# Inferring Julia types at runtime from Python objects:
#
# [Note that we sometimes use the PyFoo_Check API and sometimes we use
#  PyObject_IsInstance(o, PyFoo_Type), since sometimes the former API
#  is a macro (hence inaccessible in Julia).]

# A type-query function f(o::PyObject) returns the Julia type
# for use with the convert function, or Union{} if there isn't one.

# TODO: In Python 3.x, the BigInt check here won't work since int == long.
pyint_query(o::PyObject) = pyisinstance(o, @pyglobalobj PyInt_Type) ?
  (pyisinstance(o, @pyglobalobj :PyBool_Type) ? Bool : Int) :
  pyisinstance(o, @pyglobalobj :PyLong_Type) ? BigInt :
  pyisinstance(o, npy_integer) ? Int : Union{}

pyfloat_query(o::PyObject) = pyisinstance(o, @pyglobalobj :PyFloat_Type) ||  pyisinstance(o, npy_floating) ? Float64 : Union{}

pycomplex_query(o::PyObject) =
    pyisinstance(o, @pyglobalobj :PyComplex_Type) ||  pyisinstance(o, npy_complexfloating) ? ComplexF64 : Union{}

pystring_query(o::PyObject) = pyisinstance(o, @pyglobalobj PyString_Type) ? AbstractString : pyisinstance(o, @pyglobalobj :PyUnicode_Type) ? String : Union{}

# Given call overloading, all PyObjects are callable already, so
# we never automatically convert to Function.
pyfunction_query(o::PyObject) = Union{}

pynothing_query(o::PyObject) = o.o == pynothing[] ? Nothing : Union{}

# We refrain from converting all objects that support the mapping protocol (PyMapping_Check)
# to avoid converting types like Pandas `DataFrame` that are only lossily
# representable as a Julia dictionary (issue #376).
pydict_query(o::PyObject) = pyisinstance(o, @pyglobalobj :PyDict_Type) ? Dict{PyAny,PyAny} : Union{}

typetuple(Ts) = Tuple{Ts...}

function pysequence_query(o::PyObject)
    # pyquery(:PySequence_Check, o) always succeeds according to the docs,
    # but it seems we need to be careful; I've noticed that things like
    # scipy define "fake" sequence types with intmax lengths and other
    # problems
    if pyisinstance(o, @pyglobalobj :PyTuple_Type)
        len = @pycheckz ccall((@pysym :PySequence_Size), Int, (PyPtr,), o)
        return typetuple(pytype_query(PyObject(ccall((@pysym :PySequence_GetItem), PyPtr, (PyPtr,Int), o,i-1)), PyAny) for i = 1:len)
    elseif pyisinstance(o, pyxrange[])
        return AbstractRange
    elseif ispybytearray(o)
        return Vector{UInt8}
    elseif !haskey(o, "__array_interface__")
        # only handle PyList for now
        return pyisinstance(o, @pyglobalobj :PyList_Type) ? Array : Union{}
    else
        otypestr = get(o["__array_interface__"], PyObject, "typestr")
        typestr = convert(AbstractString, otypestr) # Could this just be String now?
        T = npy_typestrs[typestr[2:end]]
        if T == PyPtr
            T = PyObject
        end
        return Array{T}
    end
end

macro return_not_None(ex)
    quote
        T = $(esc(ex))
        if T != Union{}
            return T
        end
    end
end

const pytype_queries = Tuple{PyObject,Type}[]
"""
    pytype_mapping(pytype, jltype)

Given a Python type object `pytype`, tell PyCall to convert it to
`jltype` in `PyAny(object)` conversions.
"""
function pytype_mapping(py::PyObject, jl::Type)
    for (i,(p,j)) in enumerate(pytype_queries)
        if p == py
            pytype_queries[i] = (py,jl)
            return pytype_queries
        end
    end
    push!(pytype_queries, (py,jl))
end
"""
    pytype_query(o::PyObject, default=PyObject)

Given a Python object `o`, return the corresponding
native Julia type (defaulting to `default`) that we convert
`o` to in `PyAny(o)` conversions.
"""
function pytype_query(o::PyObject, default::TypeTuple=PyObject)
    # TODO: Use some kind of hashtable (e.g. based on PyObject_Type(o)).
    #       (A bit tricky to correctly handle Tuple and other containers.)
    @return_not_None pyint_query(o)
    pyisinstance(o, npy_bool) && return Bool
    @return_not_None pyfloat_query(o)
    @return_not_None pycomplex_query(o)
    @return_not_None pystring_query(o)
    @return_not_None pyfunction_query(o)
    @return_not_None pydate_query(o)
    @return_not_None pydict_query(o)
    @return_not_None pysequence_query(o)
    @return_not_None pyptr_query(o)
    @return_not_None pynothing_query(o)
    @return_not_None pymp_query(o)
    for (py,jl) in pytype_queries
        if pyisinstance(o, py)
            return jl
        end
    end
    return default
end

#########################################################################

function PyAny(o::PyObject)
    if ispynull(o)
        return o
    end
    try
        T = pytype_query(o)
        if T == PyObject && is_pyjlwrap(o)
            return unsafe_pyjlwrap_to_objref(o.o)
        end
        return T(o)
    catch
        pyerr_clear() # just in case
        o
    end
end

#########################################################################
