
function get!(ret::PyObject, o::PyObject, returntype::TypeTuple, k::Integer)
    if pyisinstance(o, @pyglobalobj :PyTuple_Type)
        copy!(ret, @pycheckn ccall(@pysym(:PyTuple_GetItem), PyPtr, (PyPtr, Cint), o, k))
    elseif pyisinstance(o, @pyglobalobj :PyList_Type)
        copy!(ret, @pycheckn ccall(@pysym( :PyList_GetItem), PyPtr, (PyPtr, Cint), o, k))
    else
        pydecref(ret)
        ret.o = @pycheckn ccall((@pysym :PyObject_GetItem),
                                     PyPtr, (PyPtr,PyPtr), o, PyObject(k))
    end
    convert(returntype, ret)
end

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
    pydecref(ret)
    ret.o = unsafe_load(itemsptr, k)
    pyincref_(ret.o)
    return convert(returntype, ret)
end
