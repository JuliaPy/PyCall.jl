export gettplidx, gettplidx!, getwtuple_ptr!, getwpyisinst!, unsafe_gettpl!

const MAX_PYINT = 31
const pyints = Ref{Vector{PyObject}}()
const pytuple_type_ptr = Ref{PyPtr}()

import Base: get!
function get!(ret::PyObject, o::PyObject, returntype::TypeTuple, k)
    ret.o = @pycheckn ccall((@pysym :PyObject_GetItem),
                                 PyPtr, (PyPtr,PyPtr), o, PyObject(k))
    convert(returntype, ret)
end

function getwpyisinst!(ret::PyObject, o::PyObject, returntype::TypeTuple, idx)
    if pyisinstance(o, @pyglobalobj :PyTuple_Type)
        ret.o = (@pycheckn ccall(@pysym(:PyTuple_GetItem), PyPtr, (PyPtr, Cint), o, idx))
        pyincref_(ret.o)
    else
        ret.o = @pycheckn ccall((@pysym :PyObject_GetItem),
                                     PyPtr, (PyPtr,PyPtr), o, PyObject(idx))
    end
    return convert(returntype, ret)
end

function getwtuple_ptr!(ret::PyObject, o::PyObject, returntype::TypeTuple, idx)
    pytype_ptr = unsafe_load(o.o).ob_type
    if pytype_ptr == pytuple_type_ptr[]
        ret.o = (@pycheckn ccall(@pysym(:PyTuple_GetItem), PyPtr, (PyPtr, Cint), o, idx))
        pyincref_(ret.o)
    else
        # pyerr_clear()
        ret.o = @pycheckn ccall((@pysym :PyObject_GetItem),
                                     PyPtr, (PyPtr,PyPtr), o, PyObject(idx))
    end
    return convert(returntype, ret)
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

function unsafe_gettpl!(ret::PyObject, o::PyObject, returntype::TypeTuple, idx::Int)
    pytype_ptr = unsafe_load(o.o).ob_type
    # get address of ob_item (just after the end of the struct)
    itemsptr = Base.reinterpret(Ptr{PyPtr}, o.o + sizeof(PyVar_struct))
    ret.o = unsafe_load(itemsptr, idx)
    pyincref_(ret.o)
    return convert(returntype, ret)
end
