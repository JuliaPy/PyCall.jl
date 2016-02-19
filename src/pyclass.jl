# Define Python classes out of Julia types

######################################################################
# Dispatching methods. They convert the PyObject arguments into Julia objects,
# and passes them to the Julia function `fun`

# helper for `def_py_methods`. This will call
#    fun(self_::T, args...; kwargs...)
# where `args` and `kwargs` are parsed from `args_`
function dispatch_to{T}(jl_type::Type{T}, fun::Function,
                        self_::PyPtr, args_::PyPtr)
    args = PyObject(args_)
    try
        obj = unsafe_pyjlwrap_to_objref(self_)::T
        res = fun(obj, convert(PyAny, args)...)
        return pyincref(PyObject(res)).o
    catch e
        pyraise(e)
    finally
        args.o = convert(PyPtr, C_NULL) # don't decref
    end
    return convert(PyPtr, C_NULL)
end


# Dispatching function for getters (what happens when `obj.some_field` is called
# in Python). `fun` should be a unary regular Julia function that accepts one
# object of type T, and returns that field's value (doesn't have to be an
# actual object member). 
function dispatch_get{T}(jl_type::Type{T}, fun::Function, self_::PyPtr)
    try
        obj = unsafe_pyjlwrap_to_objref(self_)::T
        res = fun(obj)
        return pyincref(PyObject(res)).o
    catch e
        pyraise(e)
    end
    return convert(PyPtr, C_NULL)
end

# The setter should accept a `new_value` argument. Return value is ignored.
function dispatch_set{T}(jl_type::Type{T}, fun::Function, self_::PyPtr,
                         value_::PyPtr)
    value = PyObject(value_)
    try
        obj = unsafe_pyjlwrap_to_objref(self_)::T
        fun(obj, convert(PyAny, value))
        return 0 # success
    catch e
        pyraise(e)
    finally
        value.o = convert(PyPtr, C_NULL) # don't decref
    end
    return -1 # failure
end



# This vector will grow on each new type (re-)definition, and never free memory.
# FIXME. I'm not sure how to detect if the corresponding Python types have
# been GC'ed.
const all_method_defs = Any[] 


"""    make_method_defs(jl_type, methods)

Create the PyMethodDef methods, stores them permanently (to prevent GC'ing),
and returns them in a Vector{PyMethodDef} """
function make_method_defs(jl_type, methods)
    method_defs = PyMethodDef[]
    for (py_name, jl_fun) in methods
        # `disp_fun` is really like the closure:
        #    (self_, args_) -> dispatch_to(jl_type, jl_fun, self_, args_)
        # but `cfunction` complains if we use that.
        disp_fun =
            @eval function $(gensym(string(jl_fun)))(self_::PyPtr, args_::PyPtr)
                dispatch_to($jl_type, $jl_fun, self_, args_)
            end
        push!(method_defs, PyMethodDef(py_name, disp_fun, METH_VARARGS))
    end
    push!(method_defs, PyMethodDef()) # sentinel

    # We have to make sure that the PyMethodDef vector isn't GC'ed by Julia, so
    # we push them onto a global stack.
    push!(all_method_defs, method_defs)
    return method_defs
end

# Similar to make_method_defs
function make_getset_defs(jl_type, getsets::Vector)
    getset_defs = PyGetSetDef[]
    for getset in getsets
        if length(getset) == 3
            (member_name, getter_fun, setter_fun) = getset
            # getters and setters have a `closure` parameter (here `_`), but it
            # was ignored in all the examples I've seen
            getter =
                @eval function $(gensym())(self_::PyPtr, _::Ptr{Void})
                    dispatch_get($jl_type, $getter_fun, self_)
                end
            setter =
                @eval function $(gensym())(self_::PyPtr, value_::PyPtr,
                                           _::Ptr{Void})
                    dispatch_set($jl_type, $setter_fun, self_, value_)
                end
            push!(getset_defs, PyGetSetDef(member_name, getter, setter))
        else
            @assert length(getset) == 2 "`getset` argument must be 2 or 3-tuple"
            (member_name, getter_fun) = getset
            getter =
                @eval function $(gensym())(self_::PyPtr, _::Ptr{Void})
                    dispatch_get($jl_type, $getter_fun, self_)
                end
            push!(getset_defs, PyGetSetDef(member_name, getter))
        end
    end
    push!(getset_defs, PyGetSetDef()) # sentinel

    push!(all_method_defs, getset_defs)    # Make sure it's not GC'ed
    return getset_defs
end

"""
Note: `some_python_obj[:x] = 10` does not call the setter at this
moment. TODO """
function def_py_methods{T}(jl_type::Type{T}, methods...;
                           base_type=pybuiltin(:object),
                           getsets=[])
    method_defs = make_method_defs(jl_type, methods)
    getset_defs = make_getset_defs(jl_type, getsets)

    # Create the Python type
    typename = jl_type.name.name::Symbol
    py_typ = pyjlwrap_type("PyCall.$typename", t -> begin 
        t.tp_getattro = @pyglobal(:PyObject_GenericGetAttr)
        t.tp_methods = pointer(method_defs)
        t.tp_getset = pointer(getset_defs)
        # Unfortunately, this supports only single-inheritance. See
        # https://docs.python.org/2/c-api/typeobj.html#c.PyTypeObject.tp_base
        t.tp_base = base_type.o # Needs pyincref?
    end)

    @eval function PyObject(obj::$T)
        pyjlwrap_new($py_typ, obj)
    end

    py_typ
end
 