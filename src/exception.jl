#########################################################################
# Wrapper around Python exceptions

struct PyError <: Exception
    msg::String # message string from Julia context, or "" if none

    # info returned by PyErr_Fetch/PyErr_Normalize
    T::PyObject
    val::PyObject
    traceback::PyObject

    # generate a PyError object.  Should normally only be called when
    # PyErr_Occurred returns non-NULL, and clears the Python error
    # indicator.
    function PyError(msg::AbstractString)
        exc = Array{PyPtr}(undef, 3)
        pexc = convert(UInt, pointer(exc))
        # equivalent of passing C pointers &exc[1], &exc[2], &exc[3]:
        ccall((@pysym :PyErr_Fetch), Cvoid, (UInt,UInt,UInt),
              pexc, pexc + sizeof(PyPtr), pexc + 2*sizeof(PyPtr))
        ccall((@pysym :PyErr_NormalizeException), Cvoid, (UInt,UInt,UInt),
              pexc, pexc + sizeof(PyPtr), pexc + 2*sizeof(PyPtr))
        new(msg, PyObject(exc[1]), PyObject(exc[2]), PyObject(exc[3]))
    end

    PyError(msg::AbstractString, e::PyError) = new(msg, e.T, e.val, e.traceback)
end

function show(io::IO, e::PyError)
    print(io, "PyError",
          isempty(e.msg) ? e.msg : string(" (",e.msg,")"),
          " ")

    if ispynull(e.T)
        println(io, "None")
    else
        println(io, pystring(e.T), "\n", pystring(e.val))
    end

    if !ispynull(e.traceback)
        o = pycall(format_traceback, PyObject, e.traceback)
        if !ispynull(o)
            for s in PyVector{AbstractString}(o)
                print(io, s)
            end
        end
    end
end

#########################################################################
# Conversion of Python exceptions into Julia exceptions

# whether a Python exception has occurred
pyerr_occurred() = ccall((@pysym :PyErr_Occurred), PyPtr, ()) != C_NULL

# call to discard Python exceptions
pyerr_clear() = ccall((@pysym :PyErr_Clear), Cvoid, ())

function pyerr_check(msg::AbstractString, val::Any)
    pyerr_occurred() && throw(PyError(msg))
    val # the val argument is there just to pass through to the return value
end

pyerr_check(msg::AbstractString) = pyerr_check(msg, nothing)
pyerr_check() = pyerr_check("")

# extract function name from ccall((@pysym :Foo)...) or ccall(:Foo,...) exprs
callsym(s::Symbol) = s
callsym(s::QuoteNode) = s.value
import Base.Meta.isexpr
callsym(ex::Expr) = isexpr(ex,:macrocall,2) ? callsym(ex.args[2]) : isexpr(ex,:ccall) ? callsym(ex.args[1]) : ex

# Macros for common pyerr_check("Foo", ccall((@pysym :Foo), ...)) pattern.
macro pycheck(ex)
    :(pyerr_check($(string(callsym(ex))), $(esc(ex))))
end

# Macros to check that ccall((@pysym :Foo), ...) returns value != bad
macro pycheckv(ex, bad)
    quote
        val = $(esc(ex))
        if val == $(esc(bad))
            # throw a PyError if available, otherwise throw ErrorException
            pyerr_check($(string(callsym(ex))))
            error($(string(callsym(ex))), " failed")
        end
        val
    end
end
macro pycheckn(ex)
    :(@pycheckv $(esc(ex)) C_NULL)
end
macro pycheckz(ex)
    :(@pycheckv $(esc(ex)) -1)
end

#########################################################################
# Mapping of Julia Exception types to Python exceptions

const pyexc = Dict{DataType, PyPtr}()
mutable struct PyIOError <: Exception end

function pyexc_initialize()
    pyexc[Exception] = @pyglobalobjptr :PyExc_RuntimeError
    pyexc[ErrorException] = @pyglobalobjptr :PyExc_RuntimeError
    pyexc[SystemError] = @pyglobalobjptr :PyExc_SystemError
    pyexc[TypeError] = @pyglobalobjptr :PyExc_TypeError
    pyexc[Meta.ParseError] = @pyglobalobjptr :PyExc_SyntaxError
    pyexc[ArgumentError] = @pyglobalobjptr :PyExc_ValueError
    pyexc[KeyError] = @pyglobalobjptr :PyExc_KeyError
    pyexc[LoadError] = @pyglobalobjptr :PyExc_ImportError
    pyexc[MethodError] = @pyglobalobjptr :PyExc_RuntimeError
    pyexc[EOFError] = @pyglobalobjptr :PyExc_EOFError
    pyexc[BoundsError] = @pyglobalobjptr :PyExc_IndexError
    pyexc[DivideError] = @pyglobalobjptr :PyExc_ZeroDivisionError
    pyexc[DomainError] = @pyglobalobjptr :PyExc_RuntimeError
    pyexc[OverflowError] = @pyglobalobjptr :PyExc_OverflowError
    pyexc[InexactError] = @pyglobalobjptr :PyExc_ArithmeticError
    pyexc[OutOfMemoryError] = @pyglobalobjptr :PyExc_MemoryError
    pyexc[StackOverflowError] = @pyglobalobjptr :PyExc_MemoryError
    pyexc[UndefRefError] = @pyglobalobjptr :PyExc_RuntimeError
    pyexc[InterruptException] = @pyglobalobjptr :PyExc_KeyboardInterrupt
    pyexc[PyIOError] = @pyglobalobjptr :PyExc_IOError
end

function pyraise(e)
    eT = typeof(e)
    pyeT = haskey(pyexc::Dict, eT) ? pyexc[eT] : pyexc[Exception]
    ccall((@pysym :PyErr_SetString), Cvoid, (PyPtr, Cstring),
          pyeT, string("Julia exception: ", e))
end

function pyraise(e::PyError)
    ccall((@pysym :PyErr_Restore), Cvoid, (PyPtr, PyPtr, PyPtr),
          e.T, e.val, e.traceback)
    e.T.o = e.val.o = e.traceback.o = C_NULL # refs were stolen
end
