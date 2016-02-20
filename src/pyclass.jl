# Define Python classes out of Julia types

using MacroTools: @capture


######################################################################
# Dispatching methods. They convert the PyObject arguments in args_ and kw_
# into Julia objects, and passes them to the Julia function `fun`

# helper for `def_py_class`. This will call
#    fun(self_::T, args...; kwargs...)
# where `args` and `kwargs` are parsed from `args_`
function dispatch_to{T}(jl_type::Type{T}, fun::Function,
                        self_::PyPtr, args_::PyPtr, kw_::PyPtr)
    # adapted from jl_Function_call
    ret_ = convert(PyPtr, C_NULL)
    args = PyObject(args_)
    try
        self = unsafe_pyjlwrap_to_objref(self_)::T
        if kw_ == C_NULL
            ret = PyObject(fun(self, convert(PyAny, args)...))
        else
            kw = PyDict{Symbol,PyAny}(PyObject(kw_))
            kwargs = [ (k,v) for (k,v) in kw ]
            ret = PyObject(fun(self, convert(PyAny, args)...; kwargs...))
        end
        ret_ = ret.o
        ret.o = convert(PyPtr, C_NULL) # don't decref
    catch e
        pyraise(e)
    finally
        args.o = convert(PyPtr, C_NULL) # don't decref
    end
    return ret_::PyPtr
end


# Dispatching function for getters (what happens when `obj.some_field` is
# called in Python). `fun` should be a Julia function that accepts (::T), and
# returns some value (it doesn't have to be an actual field of T)
function dispatch_get{T}(jl_type::Type{T}, fun::Function, self_::PyPtr)
    try
        obj = unsafe_pyjlwrap_to_objref(self_)::T
        ret = fun(obj)
        return pyincref(PyObject(ret)).o
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

######################################################################
# def_py_class definition: this is the core non-macro interface for creating
# a Python class from a Julia type.


# This vector will grow on each new type (re-)definition, and never free memory.
# FIXME maybe. I'm not sure how to detect if the corresponding Python types have
# been GC'ed.
const all_method_defs = Any[] 


"""    make_method_defs(jl_type, methods)

Create the PyMethodDef methods, stores them permanently (to prevent GC'ing),
and returns them in a Vector{PyMethodDef} """
function make_method_defs(jl_type, methods)
    method_defs = PyMethodDef[]
    for (py_name, jl_fun) in methods
        # `disp_fun` is really like the closure:
        #  (self_,args_,kw_) -> dispatch_to(jl_type, jl_fun, self_, args_, kw_)
        # but `cfunction` complains if we use that.
        disp_fun =
            @eval function $(gensym(string(jl_fun)))(self_::PyPtr, args_::PyPtr,
                                                     kw_::PyPtr)
                dispatch_to($jl_type, $jl_fun, self_, args_, kw_)
            end
        push!(method_defs, PyMethodDef(py_name, disp_fun, METH_KEYWORDS))
    end
    push!(method_defs, PyMethodDef()) # sentinel

    # We have to make sure that the PyMethodDef vector isn't GC'ed by Julia, so
    # we push them onto a global stack.
    push!(all_method_defs, method_defs)
    return method_defs
end

# Similar to make_method_defs
function make_getset_defs(jl_type, getsets::Vector)
    # getters and setters have a `closure` parameter (here `_`), but it
    # was ignored in all the examples I've seen.
    make_getter(getter_fun) = 
        @eval function $(gensym())(self_::PyPtr, _::Ptr{Void})
            dispatch_get($jl_type, $getter_fun, self_)
        end
    make_setter(setter_fun) = 
        @eval function $(gensym())(self_::PyPtr, value_::PyPtr, _::Ptr{Void})
            dispatch_set($jl_type, $setter_fun, self_, value_)
        end

    getset_defs = PyGetSetDef[]
    for getset in getsets
        # We also support getset tuples of the form
        #    ("x", some_function, nothing)
        @assert 2<=length(getset)<=3 "`getset` argument must be 2 or 3-tuple"
        if (length(getset) == 3 && getset[3] !== nothing)
            (member_name, getter_fun, setter_fun) = getset
            push!(getset_defs, PyGetSetDef(member_name, make_getter(getter_fun),
                                           make_setter(setter_fun)))
        else
            (member_name, getter_fun) = getset
            push!(getset_defs, PyGetSetDef(member_name,
                                           make_getter(getter_fun)))
        end
    end
    push!(getset_defs, PyGetSetDef()) # sentinel

    push!(all_method_defs, getset_defs)    # Make sure it's not GC'ed
    return getset_defs
end


"""    def_py_class(jl_type::Type{T}, methods...; base_class=pybuiltin(:object), getsets=[])

`def_py_class` creates a Python class by calling `pyjlwrap_new`, and defines
the corresponding `PyObject(::T)` method. `@pydef` macros expand into a call to
this function.

Arguments
---------
- `jl_type`: the Julia type (eg. Base.IO)
- `methods`: a vector of tuples `(py_name::String, jl_fun::Function)`
   py_name will be a method of the Python class, which will call `jl_function`
- `getsets`: a vector of tuples of the form
   `(py_name::String, jl_getter_fun::Function, jl_setter_fun::Function)`.
   In Python, `obj.x` will call the corresponding getter, and `obj.x = val`
   will call the setter.
- `base_class`: the Python base class to inherit from. 

Return value: the created class (::PyTypeObject)
"""
function def_py_class{T}(jl_type::Type{T}, methods...;
                         base_class=pybuiltin(:object),
                         getsets=[])
    methods = union(methods, (("_is_pydef_", io->true),)) # see is_pydef below

    method_defs = make_method_defs(jl_type, methods)
    getset_defs = make_getset_defs(jl_type, getsets)

    # Create the Python type
    typename = jl_type.name.name::Symbol
    py_typ = pyjlwrap_type("PyCall.$typename", t -> begin 
        # Note: pyjlwrap_init creates a base type for pyjlwrap_new that also
        # defines `tp_members, t.tp_repr, t.tp_hash`. I'm not sure if we should
        # bring them here, but I haven't needed them so far - cstjean
        t.tp_getattro = @pyglobal(:PyObject_GenericGetAttr)
        t.tp_methods = pointer(method_defs)
        t.tp_getset = pointer(getset_defs)
        # Necessary - otherwise we segfault on gc()
        t.tp_dealloc = pyjlwrap_dealloc_ptr
        # Unfortunately, tp_base supports only single-inheritance. See
        # https://docs.python.org/2/c-api/typeobj.html#c.PyTypeObject.tp_base
        # to add multiple-inheritance support.
        # Not sure if we need `pyincref`
        t.tp_base = pyincref(base_class).o
    end)

    @eval function PyObject(obj::$T)
        pyjlwrap_new($py_typ, obj)
    end

    py_typ
end

function is_pydef(obj::PyObject)
    # KLUDGE: before pyclass.jl, all pyjlwrap types used to inherit from
    # PyTypeObject, and this was used to detect Julia-defined types in
    # is_pyjlwrap. Since pyclass supports single-inheritance, this scheme
    # doesn't work anymore. The current solution is to add a dummy method
    # _is_py_def_ to all pyclass objects, and check for its presence
    # here. FIXME - cstjean February 2016
    try
        obj[:_is_pydef_] # triggers a KeyError if it's not a
                         # @pydef-defined object
        return true
    catch e
        if isa(e, KeyError)
            return false
        end
        rethrow()
    end
end
        

######################################################################
# @pydef macro


# Parse the `type ....` definition and returns its elements.
function parse_pydef(expr)
    if !@capture(expr, begin type type_name_ <: base_class_
                    lines__
                end end)
        @assert(@capture(expr, type type_name_
                    lines__
            end), "Malformed @pydef expression")
        base_class = pybuiltin(:object)
    end
    function_defs = Expr[] # vector of :(function ...) expressions
    methods = Tuple{AbstractString, Symbol}[] # (py_name, jl_method_name)
    getter_dict = Dict{AbstractString, Symbol}() # python_var => jl_getter_name
    setter_dict = Dict{AbstractString, Symbol}() 
    method_syms = Dict{Symbol, Symbol}() # see below
    if isa(lines[1], Expr) && lines[1].head == :block 
        # unfortunately, @capture fails to parse the `type` correctly
        lines = lines[1].args
    end
    for line in lines
        if !isa(line, LineNumberNode) && line.head != :line # need to skip those
            @assert line.head == :(=) "Malformed line: $line"
            lhs, rhs = line.args
            @assert @capture(lhs,py_f_(args__)) "Malformed left-hand-side: $lhs"
            if isa(py_f, Symbol)
                # Method definition
                # We save the gensym to support multiple dispatch
                #    readlines(io) = ...
                #    readlines(io, nlines) = ...
                # otherwise the first and second `readlines` get different
                # gensyms, and one of the two gets ignored
                jl_fun_name = get!(method_syms, py_f, gensym(py_f))
                push!(function_defs, :(function $jl_fun_name($(args...))
                    $rhs
                end))
                push!(methods, (string(py_f), jl_fun_name))
            elseif @capture(py_f, attribute_.access_)
                # Accessor (.get/.set) definition
                if access == :get
                    dict = getter_dict
                elseif access == :set!
                    dict = setter_dict
                else
                    error("Bad accessor type $access; must be either get or set!")
                end
                jl_fun_name = gensym(symbol(attribute,:_,access))
                push!(function_defs, :(function $jl_fun_name($(args...))
                    $rhs
                end))
                dict[string(attribute)] = jl_fun_name
            else
                error("Malformed line: $line")
            end
        end
    end
    @assert(isempty(setdiff(keys(setter_dict), keys(getter_dict))),
            "All .set attributes must have a .get")
    type_name, base_class, methods, getter_dict, setter_dict, function_defs
end



""" `@pydef` creates a Python class out of a Julia type. Example:

    type JuliaType
        xx
        JuliaType(xx=10) = new(xx)
    end
    
    @pyimport numpy.polynomial as P

    @pydef type JuliaType <: P.Polynomial
       py_method1(self, arg1=5) = arg1 + 20  # the right-hand-side is Julia code
       x.get(self) = self.xx
       x.set!(self, new_x::Int) = (self.xx = new_x)
    end

is equivalent to

    class JuliaType(numpy.polynomial.Polynomial):
       def __init__(self, x=10):
          self.x = 10

       def pymethod1(self, arg1):
          return arg1 + 20

Each line in a `@pydef` defines a new method or getter/setter. The right-hand
side is Julia code. For `py_method1(self, arg1) = arg1 + 20` we create a
Julia function

    function temp(self, arg1)
        arg1 + 20
    end

When Python code calls `py_method1`, its arguments are converted into
Julia objects and passed to this function `temp`. `temp`'s return value is
automatically converted back into a PyObject.

`@pydef` allows for single-inheritance of Python types. Multiple-inheritance
is not supported, but can be simulated by creating a dummy class in Python code
and inheriting from it with `@pydef`:

    class MixOfClass(BaseClass1, BaseClass):
         pass

See the PyCall usage guide on Github for more examples.
"""
macro pydef(type_expr)
    type_name, base_class, methods_, getter_dict, setter_dict, function_defs =
        parse_pydef(type_expr)
    methods = [:($py_name, $(esc(jl_fun::Symbol)))
               for (py_name, jl_fun) in methods_]
    getsets = [:($attribute,
                 $(esc(getter)),
                 $(esc(get(setter_dict, attribute, nothing))))
               for (attribute, getter) in getter_dict]
    :(begin
        $(map(esc, function_defs)...)
        def_py_class($(esc(type_name)), $(methods...);
                     base_class=$(esc(base_class)),
                     getsets=[$(getsets...)])
    end)
end
