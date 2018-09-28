# operators on Python objects

# binary operators
import Base: +,-,*,/,//,%,&,|,^,<<,>>
import Compat: ⊻
for (op,py) in ((:+,:PyNumber_Add), (:-,:PyNumber_Subtract), (:*,:PyNumber_Multiply),
                (:/,:PyNumber_TrueDivide), (:%,:PyNumber_Remainder),
                (:&,:PyNumber_And), (:|,:PyNumber_Or),
                (:<<,:PyNumber_Lshift), (:>>,:PyNumber_Rshift), (:⊻,:PyNumber_Xor))
    qpy = QuoteNode(py)
    @eval begin
        $op(a::PyObject, b::PyObject) =
            PyObject(@pycheckn @pyccall($qpy, PyPtr, (PyPtr, PyPtr), a, b))
        $op(a::PyObject, b) = $op(a, PyObject(b))
        $op(a, b::PyObject) = $op(PyObject(a), b)
    end
end

^(a::PyObject, b::PyObject) = PyObject(@pycheckn @pyccall(:PyNumber_Power, PyPtr, (PyPtr, PyPtr, PyPtr), a, b, pynothing[]))
^(a::PyObject, b) = a^PyObject(b)
^(a, b::PyObject) = PyObject(a)^b
^(a::PyObject, b::Integer) = a^PyObject(b)
Base.literal_pow(::typeof(^), x::PyObject, ::Val{p}) where {p} = x^PyObject(p)

# .+= etcetera map to in-place Python operations
for (op,py) in ((:+,:PyNumber_InPlaceAdd), (:-,:PyNumber_InPlaceSubtract), (:*,:PyNumber_InPlaceMultiply),
                (:/,:PyNumber_InPlaceTrueDivide), (:%,:PyNumber_InPlaceRemainder),
                (:&,:PyNumber_InPlaceAnd), (:|,:PyNumber_InPlaceOr),
                (:<<,:PyNumber_InPlaceLshift), (:>>,:PyNumber_InPlaceRshift), (:⊻,:PyNumber_InPlaceXor))
    qpy = QuoteNode(py)
    @eval function Base.broadcast!(::typeof($op), a::PyObject, a′::PyObject, b)
        a.o == a′.o || throw(MethodError(broadcast!, ($op, a, a', b)))
        PyObject(@pycheckn @pyccall($qpy, PyPtr, (PyPtr, PyPtr), a,PyObject(b)))
    end
end

# unary operators and functions
import Base: abs,~
for (op,py) in ((:+,:PyNumber_Positive), (:-,:PyNumber_Negative),
                (:abs,:PyNumber_Absolute), (:~, :PyNumber_Invert))
    qpy = QuoteNode(py)
    @eval $op(a::PyObject) = PyObject(@pycheckn @pyccall($qpy, PyPtr, (PyPtr,), a))
end

#########################################################################
# PyObject equality and other comparisons

# rich comparison opcodes from Python's object.h:
const Py_LT = Cint(0)
const Py_LE = Cint(1)
const Py_EQ = Cint(2)
const Py_NE = Cint(3)
const Py_GT = Cint(4)
const Py_GE = Cint(5)

import Base: <, <=, ==, !=, >, >=, isequal, isless
for (op,py) in ((:<, Py_LT), (:<=, Py_LE), (:(==), Py_EQ), (:!=, Py_NE),
                (:>, Py_GT), (:>=, Py_GE), (:isequal, Py_EQ), (:isless, Py_LT))
    @eval function $op(o1::PyObject, o2::PyObject)
        if ispynull(o1) || ispynull(o2)
            return $(py==Py_EQ || py==Py_NE || op==:isless ? :($op(o1.o, o2.o)) : false)
        elseif is_pyjlwrap(o1) && is_pyjlwrap(o2)
            return $op(unsafe_pyjlwrap_to_objref(o1.o),
                       unsafe_pyjlwrap_to_objref(o2.o))
        else
            if $(op == :isless || op == :isequal)
                return Bool(@pycheckz @pyccall(:PyObject_RichCompareBool, Cint,
                                            (PyPtr, PyPtr, Cint), o1, o2, $py))
            else # other operations may return a PyObject
                return PyAny(PyObject(@pycheckn @pyccall(:PyObject_RichCompare, PyPtr,
                                                      (PyPtr, PyPtr, Cint), o1, o2, $py)))
            end
        end
    end
    if op != :isequal
        @eval begin
            $op(o1::PyObject, o2::Any) = $op(o1, PyObject(o2))
            $op(o1::Any, o2::PyObject) = $op(PyObject(o1), o2)
        end
    end
end
# default to false since hash(x) != hash(PyObject(x)) in general
isequal(o1::PyObject, o2::Any) = !ispynull(o1) && is_pyjlwrap(o1) ? isequal(unsafe_pyjlwrap_to_objref(o1.o), o2) : false
isequal(o1::Any, o2::PyObject) = isequal(o2, o1)
