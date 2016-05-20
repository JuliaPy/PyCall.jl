# Define new Python classes from Julia:

import MacroTools: @capture

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
                      base_classes=[], getsets::Vector=[])
    # Only create new-style classes
    base_classes = union(base_classes, [pybuiltin("object")])
    new_type = pybuiltin("type")(type_name, tuple(base_classes...), Dict())
    for (py_name, jl_fun) in methods
        new_type[py_name::Symbol] = jlfun2pyfun(jl_fun::Function)
    end
    for (py_name, getter, setter) in getsets
        new_type[py_name::Symbol] = pyproperty(jlfun2pyfun(getter),
                                               jlfun2pyfun(setter))
    end
    new_type
end

######################################################################
# @pydef macro

# Helper for `parse_pydef`
# Returns (class_name::Symbol, base_classes, lines)
# where there's one `line` per method definition
function parse_pydef_toplevel(expr)
    if @capture(expr, begin type class_name_ <: base_classes_expr_
                    lines__
                end end)
        if !@capture(base_classes_expr, (base_classes__,))
            base_classes = (base_classes_expr,)
        end
    else
        @assert(@capture(expr, type class_name_
                    lines__
            end), "Malformed @pydef expression")

        base_classes = []
    end
    if isa(lines[1], Expr) && lines[1].head == :block
        # unfortunately, @capture fails to parse the type's fields correctly
        # It's been reported and fixed, we can remove this line on the next
        # MacroTools release (> v0.2)
        lines = lines[1].args
    end
    return class_name::Symbol, base_classes, lines
end

function parse_pydef(expr)
    class_name, base_classes, lines = parse_pydef_toplevel(expr)
    # Now we parse every method definition / getter / setter
    function_defs = Expr[] # vector of :(function ...) expressions
    methods = Tuple{Any, Symbol}[] # (py_name, jl_method_name)
    getter_dict = Dict{Any, Symbol}() # python_var => jl_getter_name
    setter_dict = Dict{Any, Symbol}()
    method_syms = Dict{Any, Symbol}() # see below
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
                # gensyms, and one of the two gets shadowed by the other.
                jl_fun_name = get!(method_syms, py_f, gensym(py_f))
                if py_f == :__init__
                    # __init__ must return `nothing` in Python. This is
                    # achieved by default in Python, but not so in Julia, so we
                    # special-case it for convenience.
                    rhs = :(begin $rhs; nothing end)
                end
                push!(function_defs, :(function $jl_fun_name($(args...))
                    $rhs
                end))
                push!(methods, (py_f, jl_fun_name))
            elseif @capture(py_f, attribute_.access_)
                # Accessor (.get/.set) definition
                if access == :get
                    dict = getter_dict
                elseif access == :set!
                    dict = setter_dict
                else
                    error("$access is not a valid accessor; must be either get or set!")
                end
                jl_fun_name = gensym(Symbol(attribute,:_,access))
                push!(function_defs, :(function $jl_fun_name($(args...))
                    $rhs
                end))
                dict[attribute] = jl_fun_name
            else
                error("Malformed line: $line")
            end
        end
    end
    @assert(isempty(setdiff(keys(setter_dict), keys(getter_dict))),
            "All .set attributes must have a .get")
    class_name, base_classes, methods, getter_dict, setter_dict, function_defs
end

"""
`@pydef` creates a Python class whose methods are implemented in Julia.
For instance,

    @pyimport numpy.polynomial as P
    @pydef type Doubler <: P.Polynomial
        __init__(self, x=10) = (self[:x] = x)
        my_method(self, arg1::Number) = arg1 + 20
        x2.get(self) = self[:x] * 2
        x2.set!(self, new_val) = (self[:x] = new_val / 2)
    end
    Doubler()[:x2]

is essentially equivalent to the following Python code:

    class JuliaType(numpy.polynomial.Polynomial):
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
    :(const $(esc(class_name)) = @pydef_object($(esc(class_expr))))
end

"""
`@pydef_object` is like `@pydef`, but it returns the
metaclass as a `PyObject` instead of binding it to the class name.
It's side-effect-free, except for the definition of the class methods.
"""
macro pydef_object(class_expr)
    class_name, base_classes, methods_, getter_dict, setter_dict, function_defs=
        parse_pydef(class_expr)
    methods = [:($(Expr(:quote, py_name::Symbol)), $(esc(jl_fun::Symbol)))
               for (py_name, jl_fun) in methods_]
    getsets = [:($(Expr(:quote, attribute)),
                 $(esc(getter)),
                 $(esc(get(setter_dict, attribute, nothing))))
               for (attribute, getter) in getter_dict]
    :(begin
        $(map(esc, function_defs)...)
        # This line doesn't have any side-effect, it just returns the Python
        # (meta-)class, as a PyObject
        def_py_class($(string(class_name)), [$(methods...)];
                     base_classes=[$(map(esc, base_classes)...)],
                     getsets=[$(getsets...)])
    end)
end
