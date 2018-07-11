using Base: sigatomic_begin, sigatomic_end
using PyCall: @pycheckz, TypeTuple

"""
Low-level version of `pycall(o, ...)` that always returns `PyObject`.
"""
function _pycall_legacy(o::Union{PyObject,PyPtr}, args...; kwargs...)
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
pycall_legacy(o::Union{PyObject,PyPtr}, returntype::TypeTuple, args...; kwargs...) =
    return convert(returntype, _pycall_legacy(o, args...; kwargs...)) #::returntype

pycall_legacy(o::Union{PyObject,PyPtr}, ::Type{PyAny}, args...; kwargs...) =
    return convert(PyAny, _pycall_legacy(o, args...; kwargs...))
