const Py_single_input = 256  # from Python.h
const Py_file_input = 257
const Py_eval_input = 258

const _maindict = PyDict{String,PyObject,false}(PyNULL()) # cache of __main__ module dictionary
function maindict()
    if ispynull(_maindict.o)
        _maindict.o = pyincref(@pycheckn ccall((@pysym :PyModule_GetDict), PyPtr, (PyPtr,), pyimport("__main__")))
    end
    return _maindict
end

# internal function evaluate a python string, returning PyObject, given
# Python dictionaries of global and local variables to use in the expression,
# and a current "file name" to use for stack traces
function pyeval_(s::AbstractString, globals=maindict(), locals=maindict(), input_type=Py_eval_input, fname="PyCall")
    sb = String(s) # use temp var to prevent gc before we are done with o
    sigatomic_begin()
    try
        o = PyObject(@pycheckn ccall((@pysym :Py_CompileString), PyPtr,
                                     (Cstring, Cstring, Cint),
                                     sb, fname, input_type))
        return PyObject(@pycheckn ccall((@pysym :PyEval_EvalCode),
                                         PyPtr, (PyPtr, PyPtr, PyPtr),
                                         o, globals, locals))
    finally
        sigatomic_end()
    end
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
    return convert(returntype, pyeval_(s, maindict(), locals, input_type))
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
        m = maindict()
        $assignlocals
        ret = $T(pyeval_($code_expr, m, m, $input_type, $fname))
        $removelocals
        ret
    end
end
