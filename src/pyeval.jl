const Py_single_input = 256  # from Python.h
const Py_file_input = 257
const Py_eval_input = 258

const _namespaces = Dict{Module,PyDict{String,PyObject,true}}()

pynamespace(m::Module) =
    get!(_namespaces, m) do
        if m === Main
            return PyDict{String,PyObject,true}(pyincref(@pycheckn ccall((@pysym :PyModule_GetDict), PyPtr, (PyPtr,), pyimport("__main__"))))
        else
            ns = PyDict{String,PyObject}()
            # In Python 2, it looks like `__builtin__` (w/o 's') must
            # exist at module namespace.  See also:
            # http://mail.python.org/pipermail/python-dev/2001-April/014068.html
            # https://github.com/ipython/ipython/blob/512d47340c09d184e20811ca46aaa2f862bcbafe/IPython/core/interactiveshell.py#L1295-L1299
            if pyversion < v"3"
                ns["__builtin__"] = builtin
            end
            # Following CPython implementation, we introduce
            # `__builtins__` in the namespace.  See:
            # https://docs.python.org/2/library/__builtin__.html
            # https://docs.python.org/3/library/builtins.html
            ns["__builtins__"] = builtin
            return ns
        end
    end

# internal function evaluate a python string, returning PyObject, given
# Python dictionaries of global and local variables to use in the expression,
# and a current "file name" to use for stack traces
function pyeval_(s::AbstractString, globals=pynamespace(Main), locals=pynamespace(Main),
                 input_type=Py_eval_input, fname="PyCall")
    o = PyObject(@pycheckn ccall((@pysym :Py_CompileString), PyPtr,
                                 (Cstring, Cstring, Cint),
                                 s, fname, input_type))
    ptr = disable_sigint() do
        @pycheckn ccall((@pysym :PyEval_EvalCode),
                        PyPtr, (PyPtr, PyPtr, PyPtr),
                        o, globals, locals)
    end
    return PyObject(ptr)
end

# Originally defined in https://github.com/python/cpython/blob/master/Include/compile.h
#define PyCF_DONT_IMPLY_DEDENT 0x0200
#define PyCF_ONLY_AST 0x0400
const PyCF_DONT_IMPLY_DEDENT = 0x0200
const PyCF_ONLY_AST = 0x0400

# We want to compile arbitrary python objects (e.g. `Module`) to code, but this isn't
# exposed through the C API. So instead, we use the C API to compile and evoke a _python
# call_ to the `compile` builtin (yeesh!).
function py_compile_(source, globals, locals,
                    py_input_type_str, fname, flags::Integer = 0)
    localvar = "____pycall__compile_obj_source___"
    locals[localvar] = source
    s = "compile($localvar, '$fname', '$py_input_type_str', $flags, 1)"
    pyeval_(s, globals, locals, Py_eval_input, fname)
end

#code_ast = compiler.ast_parse("2+2", filename="test.py")
function ast_parse_(s::AbstractString, globals, locals, fname="PyCall")
    # Use Py_file_input to create a Module
    code_ast = py_compile_(s, globals, locals, "exec", fname, PyCF_DONT_IMPLY_DEDENT | PyCF_ONLY_AST)
end

function pyeval_interactive_(s::AbstractString, globals=pynamespace(Main),
                             locals=pynamespace(Main), fname="PyCall")
    py_ast = pyimport("ast")

    code_ast = ast_parse_(s, globals, locals, fname)
    if code_ast.body == []
        return PyObject(nothing)
    end

    # Exec all but the last node, and eval the last node (if it's an Expr).
    # NOTE: Whereas IPython compiles the last node as 'single', which _prints_ the final
    # node _iff_ it's an Expr, we will be using 'eval' to _retrieve_ the value. However,
    # 'eval' mode is more strict, and will _only work if the input is an `ast.Expr`, so we
    # check for that here.
    if code_ast.body[end].__class__ == py_ast.Expr
        nodes_to_exec, nodes_to_eval = code_ast.body[1:end-1], code_ast.body[end].value
    else
        nodes_to_exec, nodes_to_eval = code_ast.body, nothing
    end

    function exec_node(node)
        # Compile and exec the node ("exec" requires an `ast.Module` object)
        mod = py_ast.Module([node])
        code = py_compile_(mod, globals, locals, "exec", fname, PyCF_DONT_IMPLY_DEDENT)
        eval_code(code)
    end
    function eval_node(node)
        # Compile, run, and return the result of the final node ("eval" requires an
        # `ast.Expression` object).
        mod = py_ast.Expression(node)
        code = py_compile_(mod, globals, locals, "eval", fname, PyCF_DONT_IMPLY_DEDENT)
        eval_code(code)
    end
    eval_code(code) = disable_sigint() do
        @pycheckn ccall((@pysym :PyEval_EvalCode),
                        PyPtr, (PyPtr, PyPtr, PyPtr),
                        code, globals, locals)
    end

    for n in nodes_to_exec
        # TODO: why do we iterate through each node? Why not just create one module with all
        # nodes? (This was copied from IPython...)
        exec_node(n)
    end
    ptr = nodes_to_eval === nothing ? nothing : eval_node(nodes_to_eval)

    return PyObject(ptr)
end

"""
    pyeval(s::AbstractString, returntype::TypeTuple=PyAny, locals=PyDict{AbstractString, PyObject}(),
                                input_type=Py_eval_input; kwargs...)

This evaluates `s` as a Python string and returns the result converted to `rtype` (which defaults to `PyAny`). The remaining arguments are keywords that define local variables to be used in the expression.

For example, `pyeval("x + y", x=1, y=2)` returns 3.
"""
function pyeval(s::AbstractString, returntype::TypeTuple=PyAny,
                locals=PyDict{AbstractString, PyObject}(),
                input_type=Py_eval_input; kwargs...)
    # construct deprecation warning in favor of py"..." strings
    depbuf = IOBuffer()
    q = input_type==Py_eval_input ? "\"" : "\"\"\"\n"
    qr = reverse(q)
    print(depbuf, "pyeval is deprecated.  Use ")
    if returntype == PyAny
        print(depbuf, "py$q", s, "$qr")
    elseif returntype == PyObject
        print(depbuf, "py$q", s, "$(qr)o")
    else
        print(depbuf, returntype, "(py$q", s, "$(qr)o)")
    end
    print(depbuf, " instead.")
    if !(isempty(locals) && isempty(kwargs))
        print(depbuf,  "  Use \$ interpolation to substitute Julia variables and expressions into Python.")
    end
    Base.depwarn(String(take!(depbuf)), :pyeval)

    for (k, v) in kwargs
        locals[string(k)] = v
    end
    return convert(returntype, pyeval_interactive_(s, pynamespace(Main), locals, input_type))
end

# get filename from @__FILE__ macro, which returns nothing in the REPL
make_fname(fname::AbstractString) = String(fname)
make_fname(fname::Any) = "REPL"

# a little finite-state-machine dictionary to keep track of where
# we are in Python code, since $ is ignored in string literals and comments.
#   'p' = Python code, '#' = comment, '$' = Julia interpolation
#   '"' = "..." string, '\'' = '...' string, 't' = triple-quoted string
#   '\\' = \ escape in a ' string, 'b' = \ escape in a " string, 'B' = \ in """ string
const pyFSM = Dict(
    ('p', '\'') => '\'',
    ('\'', '\'') => 'p',
    ('"', '"') => 'p',
    ('\'', '\\') => '\\', # need special handling to get out of \ mode
    ('\"', '\\') => 'b', # ...
    ('t', '\\') => 'B',  # ...
    ('p', '#') => '#',
    ('#', '\n') => 'p',
    ('p', '$') => '$',
)

# a counter so that every call to interpolate_pycode generates
# unique local-variable names.
const _localvar_counter = Ref(0)

# Given Python code, return (newcode, locals), where
# locals is a Dict of identifier string => expr for expressions
# that should be evaluated and assigned to Python identifiers
# for use in newcode, to represent $... interpolation in code.
# For $$ interpolation, which pastes a string directly into
# the Python code, locals contains position -> expression, where
# position is the index in the buffer string where the result
# of the expression should be inserted as a string.
function interpolate_pycode(code::AbstractString)
    buf = IOBuffer() # buffer to hold new/processed Python code
    state = 'p' # Python code.
    i = 1 # position in code
    locals = Dict{Union{String,Int},Any}()
    numlocals = 0
    localprefix = "__julia_localvar_$(_localvar_counter[])_"
    _localvar_counter[] += 1
    while i <= lastindex(code)
        c = code[i]
        newstate = get(pyFSM, (state, c), '?')
        if newstate == '$' # Julia expression to interpolate
            i += 1
            i > lastindex(code) && error("unexpected end of string after \$")
            interp_literal = false
            if code[i] == '$' # $$foo pastes the string foo into the Python code
                i += 1
                interp_literal = true
            end
            expr, i = Meta.parse(code, i, greedy=false)
            if interp_literal
                # need to save both the expression and the position
                # in the string where it should be interpolated
                locals[position(buf)+1] = expr
            else
                numlocals += 1
                localvar = string(localprefix, numlocals)
                locals[localvar] = expr
                print(buf, localvar)
            end
        else
            if newstate == '?' # cases that need special handling
                if state == 'p'
                    if c == '"' # either " or """ string
                        if i + 2 <= lastindex(code) && code[i+1] == '"' && code[i+2] == '"'
                            i = i + 2
                            newstate = 't'
                        else
                            newstate = '"'
                        end
                    else
                        newstate = 'p'
                    end
                elseif state in ('#', '"', '\'')
                    newstate = state
                elseif state == '\\'
                    newstate = '\''
                elseif state == 'b'
                    newstate = '"'
                elseif state == 'B'
                    newstate = 't'
                elseif state == 't'
                    if c == '"' && i + 2 <= lastindex(code) && code[i+1] == '"' && code[i+2] == '"'
                        i = i + 2
                        newstate = 'p'
                    end
                end
            end
            print(buf, c)
            state = newstate
            i = nextind(code, i)
        end
    end
    return String(take!(buf)), locals
end

"""
    py".....python code....."

Evaluate the given Python code string in the main Python module.

If the string is a single line (no newlines), then the Python
expression is evaluated and the result is returned.
If the string is multiple lines (contains a newline), then the Python
code is compiled and evaluated in the `__main__` Python module
and nothing is returned.

If the `o` option is appended to the string, as in `py"..."o`, then the
return value is an unconverted `PyObject`; otherwise, it is
automatically converted to a native Julia type if possible.

Any `\$var` or `\$(expr)` expressions that appear in the Python code
(except in comments or string literals) are evaluated in Julia
and passed to Python via auto-generated global variables. This
allows you to "interpolate" Julia values into Python code.

Similarly, ny `\$\$var` or `\$\$(expr)` expressions in the Python code
are evaluated in Julia, converted to strings via `string`, and are
pasted into the Python code.   This allows you to evaluate code
where the code itself is generated by a Julia expression.
"""
macro py_str(code, options...)
    T = length(options) == 1 && 'o' in options[1] ? PyObject : PyAny
    code, locals = interpolate_pycode(code)
    input_type = '\n' in code ? Py_file_input : Py_eval_input
    fname = make_fname(@__FILE__)
    assignlocals = Expr(:block, [(isa(v,String) ?
                                  :(m[$v] = PyObject($(esc(ex)))) :
                                  nothing) for (v,ex) in locals]...)
    code_expr = Expr(:call, esc(:(Base.string)))
    i0 = firstindex(code)
    for i in sort!(collect(filter(k -> isa(k,Integer), keys(locals))))
        push!(code_expr.args, code[i0:prevind(code,i)], esc(locals[i]))
        i0 = i
    end
    push!(code_expr.args, code[i0:lastindex(code)])
    if input_type == Py_eval_input
        removelocals = Expr(:block, [:(delete!(m, $v)) for v in keys(locals)]...)
    else
        # if we are evaluating multi-line input, then it is not
        # safe to remove the local variables, because they might be referred
        # to in Python function definitions etc. that will be called later.
        removelocals = nothing
    end
    quote
        m = pynamespace($__module__)
        $assignlocals
        ret = $T(pyeval_interactive_($code_expr, m, m, $fname))
        $removelocals
        ret
    end
end
