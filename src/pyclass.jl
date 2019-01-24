# Define new Python classes from Julia.
#############################################################################

import MacroTools: @capture, splitdef, combinedef

######################################################################
# def_py_class definition: this is the core non-macro interface for creating
# a Python class from a Julia type.

"""
    def_py_class(type_name::AbstractString, methods::Vector;
                 base_classes=[], getsets::Vector=[])

`def_py_class` creates a Python class whose methods are implemented in Julia.
`@pydef` macros expand into a call to this function.

Arguments
---------
- `methods`: a vector of tuples `(py_name::String, jl_fun::Function)`
   `py_name` will be a method of the Python class, which will call `jl_fun`
- `base_classes`: the Python base classes to inherit from.

Return value: the created class (`::PyTypeObject`)
"""
function def_py_class(type_name::AbstractString, methods::Vector;
                      base_classes=[], getsets::Vector=[], class_vars=[])
    # Only create new-style classes
    base_classes = union(base_classes, [pybuiltin("object")])
    methods = Dict(py_name => jlfun2pyfun(jl_fun::Function)
                   for (py_name::Symbol, jl_fun) in methods)
    getter_setters = Dict(py_name => pyproperty(jlfun2pyfun(getter),
                                                jlfun2pyfun(setter))
                          for (py_name::Symbol, getter, setter) in getsets)
    return pybuiltin("type")(type_name, tuple(base_classes...),
                             merge(methods, getter_setters, Dict(class_vars)))
end

######################################################################
# @pydef macro

# Helper for `parse_pydef`
# Returns (class_name::Symbol, base_classes, lines)
# where there's one `line` per method definition
function parse_pydef_toplevel(expr)
    if @capture(expr, begin mutable struct class_name_ <: base_classes_expr_
                    lines__
                end end)
        if !@capture(base_classes_expr, (base_classes__,))
            base_classes = (base_classes_expr,)
        end
    else
        @assert(@capture(expr, mutable struct class_name_
                    lines__
            end), "Malformed @pydef expression")

        base_classes = []
    end
    return class_name::Symbol, base_classes, lines
end

# From MacroTools
function isfunction(expr)
    @capture(MacroTools.longdef1(expr), function (fcall_ | fcall_) body_ end)
end

function parse_pydef(expr)
    class_name, base_classes, lines = parse_pydef_toplevel(expr)
    # Now we parse every method definition / getter / setter
    function_defs = Expr[] # vector of :(function ...) expressions
    methods = Tuple{Any, Symbol}[] # (py_name, jl_method_name)
    getter_dict = Dict{Any, Symbol}() # python_var => jl_getter_name
    setter_dict = Dict{Any, Symbol}()
    method_syms = Dict{Any, Symbol}() # see below
    class_vars = Dict{Symbol, Any}()
    for line in lines
        line isa LineNumberNode && continue
        line isa Expr || error("Malformed line: $line")
        line.head == :line && continue
        if isfunction(line)
            def_dict = splitdef(line)
            py_f = def_dict[:name]
            # The dictionary of the new Julia-side definition.
            jl_def_dict = copy(def_dict)
            if isa(py_f, Symbol)
                # Method definition
                # We save the gensym to support multiple dispatch
                #    readlines(io) = ...
                #    readlines(io, nlines) = ...
                # otherwise the first and second `readlines` get different
                # gensyms, and one of the two gets shadowed by the other.
                jl_def_dict[:name] = get!(method_syms, py_f, gensym(py_f))
                if py_f == :__init__
                    # __init__ must return `nothing` in Python. This is
                    # achieved by default in Python, but not so in Julia, so we
                    # special-case it for convenience.
                    jl_def_dict[:body] = :(begin $(def_dict[:body]); nothing end)
                end
                push!(methods, (py_f, jl_def_dict[:name]))
            elseif @capture(py_f, attribute_.access_)
                # Accessor (.get/.set) definition
                if access == :get
                    dict = getter_dict
                elseif access == :set!
                    dict = setter_dict
                else
                    error("$access is not a valid accessor; must be either get or set!")
                end
                dict[attribute] = jl_def_dict[:name] = gensym(Symbol(attribute,:_,access))
            else
                error("Malformed line: $line")
            end
            push!(function_defs, combinedef(jl_def_dict))
        elseif line.head == :(=) # Non function assignment
            class_vars[line.args[1]] = line.args[2]
        else
            error("Malformed line: $line")
        end
    end
    @assert(isempty(setdiff(keys(setter_dict), keys(getter_dict))),
            "All .set attributes must have a .get")
    return (
        class_name,
        base_classes,
        methods,
        getter_dict,
        setter_dict,
        function_defs,
        class_vars
    )
end

"""
`@pydef` creates a Python class whose methods are implemented in Julia.
For instance,

    P = pyimport("numpy.polynomial")
    @pydef type Doubler <: P.Polynomial
        __init__(self, x=10) = (self.x = x)
        my_method(self, arg1::Number) = arg1 + 20
        x2.get(self) = self.x * 2
        x2.set!(self, new_val) = (self.x = new_val / 2)
    end
    Doubler().x2

is essentially equivalent to the following Python code:

    class Doubler(numpy.polynomial.Polynomial):
        def __init__(self, x=10):
            self.x = x
        def my_method(self, arg1): return arg1 + 20
        @property
        def x2(self): return self.x * 2
        @x2.setter
        def x2(self, new_val):
            self.x = new_val / 2
    Doubler().x2

The method arguments and return values are automatically converted between Julia and Python. All Python
special methods are supported (`__len__`, `__add__`, etc.)

`@pydef` allows for multiple inheritance of Python types:

    @pydef type SomeType <: (BaseClass1, BaseClass2)
        ...
    end

Multiple dispatch works, too:

    x2.set!(self, new_x::Int) = ...
    x2.set!(self, new_x::Float64) = ...
"""
macro pydef(class_expr)
    class_name, _, _ = parse_pydef_toplevel(class_expr)
    esc(:($class_name = $PyCall.@pydef_object($class_expr)))
end

"""
`@pydef_object` is like `@pydef`, but it returns the
metaclass as a `PyObject` instead of binding it to the class name.
It's side-effect-free, except for the definition of the class methods.
"""
macro pydef_object(class_expr)
    class_name,
    base_classes,
    methods_,
    getter_dict,
    setter_dict,
    function_defs,
    class_vars =
        parse_pydef(class_expr)
    methods = [:($(Expr(:quote, py_name::Symbol)), $(esc(jl_fun::Symbol)))
               for (py_name, jl_fun) in methods_]
    getsets = [:($(Expr(:quote, attribute)),
                 $(esc(getter)),
                 $(esc(get(setter_dict, attribute, nothing))))
               for (attribute, getter) in getter_dict]
    class_var_pairs = [
        :($(Expr(:quote, py_name)), $(esc(val_expr)))
        for (py_name, val_expr) in class_vars
    ]
    :(begin
        $(map(esc, function_defs)...)
        # This line doesn't have any side-effect, it just returns the Python
        # (meta-)class, as a PyObject
        def_py_class($(string(class_name)), [$(methods...)];
                     base_classes=[$(map(esc, base_classes)...)],
                     getsets=[$(getsets...)], class_vars = [$(class_var_pairs...)])
    end)
end
