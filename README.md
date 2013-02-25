# Calling Python functions from the Julia language

This package provides a `@pyimport` macro that mimics a Python
`import` statement: it imports a Python module and provides Julia
wrappers for all of the functions and constants therein, including
automatic conversion of types between Julia and Python.

It also provides facilities for lower-level manipulation of Python
objects, including a `PyObject` type for opaque Python objects and a
`pycall` function (similar in spirit to Julia's `ccall` function) to
call Python functions from the Julia language with type conversions.

## Installation

Within Julia, just use the package manager to run `Pkg.add("PyCall")` to
install the files.

The latest development version of PyCall is avalable from
<https://github.com/stevengj/PyCall.jl>.  If you want to switch to
this after installing the package, `cd ~/.julia/PyCall` and `git pull
git://github.com/stevengj/PyCall.jl master`.

## Usage

Here is a simple example to call Python's `math.sin` function and
compare it to the built-in Julia `sin`:

    using PyCall
    @pyimport math
    math.sin(math.pi / 4) - sin(pi / 4)  # returns 0.0

Type conversions are automatically performed for numeric, boolean, and
string types, along with tuples, arrays/lists, and dictionaries of
these types.  Python functions can be converted to Julia functions but
not vice-versa.  Other types are supported via the generic PyObject type, below.

Python submodules must be imported by a separate `@pyimport` call, and
in this case you must supply an identifier to to use in Julia.  For example

    @pyimport numpy.random as nr
    nr.rand(3,4)

Multidimensional arrays rely on the NumPy array interface for
conversions between Python and Julia.  By default, they are passed
from Julia to Python without making a copy, but from Python to Julia a
copy is made; no-copy conversion of Python to Julia arrays can be achieved
with the `PyArray` type below.

Keyword arguments can be passed as the *last* argument of a function,
prefixed by the `@pykw` macro, and separated by *spaces*.  For example,
matplotlib's [pylab](http://matplotlib.org/) uses keyword arguments to specify
plot options, and this functionality is accessed from Julia by:

    @pyimport pylab
    x = linspace(0,2*pi,1000); y = sin(3*x + 4*cos(2*x));
    pylab.plot(x, y, @pykw color="red" linewidth=2.0 linestyle="--")
    pylab.show() 

The biggest diffence from Python is that object attributes/members are
accessed with `o[:attribute]` rather than `o.attribute`.  (This is because Julia
does not permit overloading the `.` operator yet.)  See also the section on
`PyObject` below, as well as the `pywrap` function to create anonymous
composite types that simulate `.` access (this is what `@pyimport` does).
For example, using [Biopython](http://biopython.org/wiki/Seq) we can do:

    @pyimport Bio.Seq as s
    @pyimport Bio.Alphabet as a
    my_dna = s.Seq("AGTACACTGGT", a.generic_dna)
    my_dna[:find]("ACT")

whereas in Python the last step would have been `mydna.find("ACT")`.

## Python object interfaces

The `@pyimport` macro is built on top of several routines for
manipulating Python objects in Julia, via a type `PyObject` described
below.  These can be used to have greater control over the types and
data passed between Julia and Python, as well as to access additional
Python functionality (especially in conjunction with the low-level interfaces
described later).

### Types

#### PyObject

The PyCall module also provides a new type `PyObject` (a wrapper around
`PyObject*` in Python's C API) representing a reference to a Python object.

Constructors `PyObject(o)` are provided for a number of Julia types,
and PyCall also supplies `convert(T, o::PyObject)` to convert
PyObjects back into Julia types `T`.  Currently, the only types
supported are numbers (integer, real, and complex), booleans, and
strings, along with tuples and arrays/lists thereof, but more are planned.

Given a `o::PyObject`, `o[:attribute]` is equivalent to `o.attribute`
in Python, with automatic type conversion.  To get an attribute as a
`PyObject` without type conversion, do `o["attribute"]` instead.

#### PyArray

Multidimensional NumPy arrays (`ndarray`) are supported and can be
converted to the native Julia `Array` type, which makes a copy of the data.

Alternatively, the PyCall module also provides a new type `PyArray` (a
subclass of `AbstractArray`) which implements a no-copy wrapper around
a NumPy array (currently of numeric types or objects only).  Just use
`PyArray` as the return type of a `pycall` returning an `ndarray`, or
call `PyArray(o::PyObject)` on an `ndarray` object `o`.  (Technically,
a `PyArray` works for any Python object that uses the NumPy array
interface to provide a data pointer and shape information.)  

Conversely, when passing arrays *to* Python, Julia `Array` types are
converted to `PyObject` types *without* making a copy via NumPy,
e.g. when passed as `pycall` arguments. **Warning:** If Python creates
a new reference to an `Array` object and returns it from `pycall`, you
*must* ensure that the original `Array` object still exists (i.e., is not
garbage collected) as long as any such "hidden" Python references
exist.

#### PyDict

Similar to `PyArray`, PyCall also provides a type `PyDict` (a subclass
of `Association`) that implements a no-copy wrapper around a Python
dictionary (or any object implementing the mapping protocol).  Just
use `PyDict` as the return type of a `pycall` returning a dictionary,
or call `PyDict(o::PyObject)` on a dictionary` object `o`.  By
default, a `PyDict` is an `Any => Any` dictionary (or actually `PyAny
=> PyAny`) that performs runtime type inference, but if your Python
dictionary has known, fixed types you can insteady use `PyDict{K,V}` given
the key and value types `K` and `V` respectively.

Currently, passing Julia dictionaries to Python makes a copy of the Julia
dictionary.

#### PyAny

The `PyAny` type is used in conversions to tell PyCall to detect the
Python type at runtime and convert to the corresponding native Julia
type.  That is, `pycall(func, PyAny, ...)` and `convert(PyAny,
o::PyObject)` both automatically convert their result to a native
Julia type (if possible).   This is convenient, but will lead
to slightly worse performance (due to the overhead of runtime type-checking
and the fact that the Julia JIT compiler can no longer infer the type).

### Calling Python

In most cases, the `@pyimport` macro automatically makes the
appropriate type conversions to Julia types based on runtime
inspection of the Python objects.  However greater control over these
type conversions (e.g. to use a no-copy `PyArray` for a Python
multidimensional array rather than copying to an `Array`) can be
achieved by using the lower-level functions below.  Using `pycall` in
cases where the Python return type is known can also improve
performance, both by eliminating the overhead of runtime type inference
and also by providing more type information to the Julia compiler.

* `pycall(function::PyObject, returntype::Type, args...)`.   Call the given 
  Python `function` (typically looked up from a module) with the given
  `args...` (of standard Julia types which are converted automatically to
  the corresponding Python types if possible), converting the return value
  to `returntype` (use a `returntype` of `PyObject` to return the unconverted
  Python object reference, or of `PyAny` to request an automated conversion).

* `pyimport(s)`: Import the Python module `s` (a string or symbol) and
  return a pointer to it (a `PyObject`).   Functions or other symbols
  in the module may then be looked up by `s[name]` where `name` is a string
  or symbol (`s[name]` also returns a `PyObject`).  Unlike the `@pyimport`
  macro, this does not define a structure type and module members cannot
  be accessed with `s.name` (the `s[name]` syntax can also be used
  with `@pyimport` modules in order to obtain the raw `PyObject` members).

* `pybuiltin(s)`: Look up `s` (a string or symbol) among the global Python
  builtins, returning a `PyObject`

* `pywrap(o::PyObject)` returns a wrapper `w` that is an anonymous
  composite type (a subclass of `PyWrapper`) which provides (read)
  access to converted versions of `o`'s members as `w.member`.  (For
  example, `@pyimport module as name` is equivalent to `name =
  pywrap(pyimport("module"))`.)

### Initialization

By default, whenever you call any of the high-level PyCall routines
above, the Python interpreter (corresponding to the `python`
executable name) is initialized and remains in memory until Julia
exits.

However, you may want to modify this behavior to change the default
Python version, to call low-level Python functions directly via
`ccall`, or to free the memory consumed by Python.  This can be
accomplished using:

* `pyinitialize(s::String)`: Initialize the Python interpreter using
  the Python libraries corresponding to the `python` executable given
  by the argument `s`.  Calling `pyinitialize()` defaults to
  `pyinitialize("python")`, but you may need to change this to use a
  different Python version.  The `pyinitialize` function *must* be
  called before you can call any low-level Python functions (via
  `ccall`), but it is called automatically as needed when you use the
  higher-level functions above.  It is safe to call this function more
  than once; subsequent calls will do nothing.

* `pyfinalize()`: End the Python interpreter and free all associated
  memory.  After this function is called, you *may no longer restart
  Python* by calling `pyinitialize` again (an exception will be
  thrown).  The reason is that some Python modules (e.g. numpy) crash
  if their initialization routine is called more than once.
  Subsequent calls to `pyfinalize` do nothing.  You must *not* try
  to access any Python functions or data (that has not been *copied*
  to native Julia types) after `pyfinalize` runs!

* The Python version number is returned by `pyversion()`, which returns
  Julia's native `VersionNumber` type.

### Low-level Python API access

If you want to call low-level functions in the Python C API, you can
do so using `ccall`.  Just remember to call `pyinitialize()` first, and:

* Use `pyfunc(func::Symbol)` to get a function pointer to pass to `ccall`
  given a symbol `func` in the Python API.  e.g. you can call `int Py_IsInitialized()` by `ccall(pyfunc(:Py_IsInitialized), Int32, ())`.

* PyCall defines the typealias `PyPtr` for `PythonObject*` argument types,
  and `PythonObject` (see above) arguments are correctly converted to this
  type.  `PythonObject(p::PyPtr)` creates a Julia wrapper around a
  `PyPtr` return value.

* Use `PythonObject` and the `convert` routines mentioned above to convert
  Julia types to/from `PythonObject*` references.

* If a new reference is returned by a Python function, immediately
  convert the `PyPtr` return values to `PythonObject` objects in order to
  have their Python reference counts decremented when the object is
  garbage collected in Julia.  i.e. `PythonObject(ccall(func, PyPtr, ...))`.
  **Important**: for Python routines that return a borrowed reference,
  you should instead do `pyincref(PyObject(...))` to obtain a new
  reference.

* You can call `pyincref(o::PyObject)` and `pydecref(o::PyObject)` to
  manually increment/decrement the reference count.  This is sometimes
  needed when low-level functions steal a reference or return a borrowed one.

* The function `pyerr_check(msg::String)` can be used to check if a
  Python exception was thrown, and throw a Julia exception (which includes
  both `msg` and the Python exception object) if so.  The Python 
  exception status may be cleared by calling `pyerr_clear()`.

* The function `pytype_query(o::PyObject)` returns a native Julia
  type that `o` can be converted into, if possible, or `PyObject` if not.

* `pyisinstance(o::PyObject, t::Symbol)` can be used to query whether
  `o` is of a given Python type (where `t` is the identifier of a global
  `PyTypeObject` in the Python C API), e.g. `pyisinstance(o, :PyDict_Type)`
  checks whether `o` is a Python dictionary.  Alternatively,
  `pyisinstance(o::PyObject, t::PyObject)` performs the same check
  given a Python type object `t`.

## Work in Progress

* Conversions for many more types (set, range, xrange, etc.). 

* Support for Julia callback functions passed to Python.

* A PyList type for no-copy sharing of Python sequence objects.

* Caching of `pyfunc` results, following [this suggestion](https://github.com/stevengj/PyCall.jl/pull/2).

## Author

This package was written by [Steven G. Johnson](http://math.mit.edu/~stevenj/).
