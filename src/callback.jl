# Passing Julia callback functions to Python routines.
#
# Note that this will typically involve two functions: the
# desired Julia function/closure, and a top-level C-callable
# wrapper function used with PyCFunction_NewEx -- the latter
# is called from Python and calls the former as needed.

################################################################
# mirror of Python API types and constants from methodobject.h

type PyMethodDef
    ml_name::Ptr{Uint8}
    ml_meth::Ptr{Void}
    ml_flags::Cint
    ml_doc::Ptr{Uint8} # may be NULL
end

# A PyCFunction is a C function of the form
#     PyObject *func(PyObject *self, PyObject *args)
# The first parameter is the "self" function for method, or 
# for module functions it is the module object.  The second
# parameter is either a tuple of args (for METH_VARARGS),
# a single arg (for METH_O), or NULL (for METH_NOARGS).  func
# must return non-NULL (Py_None is okay) unless there was an
# error, in which case an exception must have been set.

# ml_flags should be one of:
const METH_VARARGS = 0x0001 # args are a tuple of arguments
const METH_NOARGS = 0x0004  # no arguments (NULL argument pointer)
const METH_O = 0x0008       # single argument (not wrapped in tuple)

################################################################

# Define a Python method/function object from f(PyPtr,PyPtr)::PyPtr.
# Requires f to be a top-level function.
function pymethod(f::Function, name::String, flags::Integer)
    # Python expects the PyMethodDef structure to be a *constant*,
    # and the strings pointed to therein must also be constants,
    # so we define anonymous globals to hold these
    def = gensym("PyMethodDef")
    defname = gensym("PyMethodDef_ml_name")
    @eval const $defname = bytestring($name)
    @eval const $def = PyMethodDef(convert(Ptr{Uint8}, $defname),
                                   $(cfunction(f, PyPtr, (PyPtr,PyPtr))),
                                   convert(Cint, $flags),
                                   convert(Ptr{Uint8}, C_NULL))
    PyObject(@pycheckn ccall(pyfunc(:PyCFunction_NewEx), PyPtr,
                             (Ptr{PyMethodDef}, Ptr{Void}, Ptr{Void}),
                             &eval(def), C_NULL, C_NULL))
end

################################################################
