#########################################################################
# Once Julia lets us overload ".", we will use [] to access items, but
# for now we can define "get".

###############################
# get with k<:Any and a default
###############################
function get!(ret::PyObject, o::PyObject, returntype::TypeTuple, k, default)
    r = ccall((@pysym :PyObject_GetItem), PyPtr, (PyPtr,PyPtr), o, PyObject(k))
    if r == C_NULL
        pyerr_clear()
        default
    else
        convert(returntype, PyObject(r))
    end
end

get(o::PyObject, returntype::TypeTuple, k, default) =
  get!(PyNULL(), o, returntype, k, default)

# returntype defaults to PyAny
get!(ret::PyObject, o::PyObject, k, default) = get!(ret, o, PyAny, k, default)
get(o::PyObject, k, default) = get(o, PyAny, k, default)

###############################
# get with k<:Any
###############################
function get!(ret::PyObject, o::PyObject, returntype::TypeTuple, k)
    pydecref(ret)
    ret.o = @pycheckn ccall((@pysym :PyObject_GetItem),
                                 PyPtr, (PyPtr,PyPtr), o, PyObject(k))
    return convert(returntype, ret)
end

get(o::PyObject, returntype::TypeTuple, k) = get!(PyNULL(), o, returntype, k)

# returntype defaults to PyAny
get!(ret::PyObject, o::PyObject, k) = get!(ret, o, PyAny, k)
get(o::PyObject, k) = get(o, PyAny, k)

###############################
# get with k<:Integer
###############################
function get!(ret::PyObject, o::PyObject, returntype::TypeTuple, k::Integer)
    if pyisinstance(o, @pyglobalobj :PyTuple_Type)
        copy!(ret, @pycheckn ccall(@pysym(:PyTuple_GetItem), PyPtr, (PyPtr, Cint), o, k))
    elseif pyisinstance(o, @pyglobalobj :PyList_Type)
        copy!(ret, @pycheckn ccall(@pysym( :PyList_GetItem), PyPtr, (PyPtr, Cint), o, k))
    else
        return get!(ret, o, returntype, PyObject(k))
    end
    return convert(returntype, ret)
end

get(o::PyObject, returntype::TypeTuple, k::Integer) =
    get!(PyNULL(), o, returntype, k)

# default to PyObject(k) methods for no returntype, and default variants
get!(ret::PyObject, o::PyObject, returntype::TypeTuple, k::Integer, default) =
    get!(ret, o, returntype, PyObject(k), default)

get!(ret::PyObject, o::PyObject, k::Integer) = get!(ret, o, PyObject(k))

get!(ret::PyObject, o::PyObject, k::Integer, default) =
    get!(ret, o, PyObject(k), default)

###############################
# unsafe_gettpl!
###############################

# struct PyTuple_struct
# refs:
#   https://github.com/python/cpython/blob/da1734c58d2f97387ccc9676074717d38b044128/Include/object.h#L106-L115
#   https://github.com/python/cpython/blob/da1734c58d2f97387ccc9676074717d38b044128/Include/tupleobject.h#L25-L33
struct PyVar_struct
    ob_refcnt::Int
    ob_type::Ptr{Cvoid}
    ob_size::Int
    # ob_item::Ptr{PyPtr}
end

function unsafe_gettpl!(ret::PyObject, o::PyObject, returntype::TypeTuple, k::Int)
    pytype_ptr = unsafe_load(o.o).ob_type
    # get address of ob_item (just after the end of the struct)
    itemsptr = Base.reinterpret(Ptr{PyPtr}, o.o + sizeof(PyVar_struct))
    copy!(ret, unsafe_load(itemsptr, k+1)) # unsafe_load is 1-based
    return convert(returntype, ret)
end
