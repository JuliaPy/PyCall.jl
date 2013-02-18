# Calling Python functions from the Julia language

This package provides a `pycall` function (similar in spirit to
Julia's `ccall` function) to call Python functions from the Julia
language, automatically converting types etcetera.

## Work in Progress

PyCall is currently a proof-of-concept and work in progress.  Much
basic functionality works, but major TODO items are:

* Automatic type inference of Python return values (currently, you
  must specify this manually, which leads to better compiled code
  but is inconvenient).

* More error checking and conversion of Python exceptions to Julia exceptions.

* Conversions for many more types (set, range, xrange, etc.).  Callback
  functions.

* Syntactic sugar.

## Installation

Until this package stabilizes and is added to Julia's global
[METADATA.jl](https://github.com/JuliaLang/METADATA.jl) database, you should
do

    cd ~/.julia
    git clone https://github.com/stevengj/PyCall.jl PyCall

to fetch the latest version of the package and install it in your
`.julia` directory (or somewhere else if you prefer).

## Usage

Here is a simple example to call Python's `math.sin` function:

    using PyCall
    math = pyimport("math") # import the Python math module
    pycall(math["sin"], Float64, 3.0) - sin(3.0) # returns 0.0

Note that `math["sin"]` looks up the `sin` function in the Python
`math` module, and is the equivalent of `math.sin` in Python.

Like `ccall`, we must tell `pycall` the return type we wish in Julia.
In principle, this could be determined dynamically at runtime but that
is not currently implemented, in part because that would defeat
efficient compilation in Julia (since the Julia compiler would be
unable to determine the return type at compile-time).  On the other
hand, the argument types need not be specified explicitly; they will
be determined from the types of the Julia arguments, which will be
converted into the corresponding Python types.  For example:

    pycall(math["sin"], Float64, 3) - sin(3.0)

also works and also returns `0.0` since Python's `math.sin` function accepts
integer arguments as well as floating-point arguments.

Currently, numeric, boolean, and string types, along with tuples and
arrays/lists thereof, are supported, with more planned.

You can also look up other names in a module, and use `convert` to
convert them to Julia types, e.g.

    convert(Float64, math["pi"])

returns the numeric value of &pi; from Python's `math.pi`.

## Reference

The PyCall module supplies several subroutines, types, and conversion routines
to simplify calling Python code.

### Calling Python

* `pyimport(s)`: Import the Python module `s` (a string or symbol) and
  return a pointer to it (a `PyObject`).   Functions or other symbols
  in the module may then be looked up by `s[name]` where `name` is a string
  or symbol (`s[name]` also returns a `PyObject`).

* `pycall(function::PyObject, returntype::Type, args...)`.   Call the given 
  Python `function` (typically looked up from a module) with the given
  `args...` (of standard Julia types which are converted automatically to
  the corresponding Python types if possible), converting the return value
  to `returntype`.

* `pyglobal(s)`: Look up `s` (a string or symbol) in the global Python
  namespace.

* Note that \_\_builtin\_\_ module gives access to all top-level python functions:
  ```julia 
  julia> using PyCall

  julia> pybuiltin = pyimport("__builtin__")
  PyObject <module '__builtin__' (built-in)>

  julia> pyfile = pybuiltin["file"]
  PyObject <type 'file'>
  ```

### Types

The PyCall module also provides a new type `PyObject` (a wrapper around
`PyObject*` in Python's C API) representing a reference to a Python object.

Constructors `PyObject(o)` are provided for a number of Julia types,
and PyCall also supplies `convert(T, o::PyObject)` to convert
PyObjects back into Julia types `T`.  Currently, the only types
supported are numbers (integer, real, and complex), booleans, and
strings, along with tuples and arrays/lists thereof, but more are planned.

### Initialization

By default, whenever you call `pyimport`, `pycall`, or `pyglobal`, the
Python interpreter (corresponding to the `python` executable name) is
initialized and remains in memory until Julia exits.  However, you may
want to modify this behavior to change the default Python version, to
call low-level Python functions or create `PyObject`s before calling
`pyimport` etcetera, or to free the memory consumed by Python.  This
can be accomplished using:

* `pyinitialize(s::String)`: Initialize the Python interpreter using
  the Python libraries corresponding to the `python` executable given
  by the argument `s`.  Calling `pyinitialize()` defaults to
  `pyinitialize("python")`, but you may need to change this to use a
  different Python version.   The `pyinitialize` function *must* be
  called before you can call any Python functions, but it is called
  automatically if necessary by `pyimport`, `pycall`, and `pyglobal`.
  It is safe to call this function more than once; subsequent calls will
  do nothing (until `pyfinalize` is called).

* `pyfinalize()`: End the Python interpreter and free all associated memory.
  After this function is called, you may restart the Python interpreter
  by calling `pyinitialize` again.  It is safe to call `pyfinalize` more
  than once (subsequent calls do nothing).   You must *not* have any
  remaining variables referencing `PyObject` types when `pyfinalize` runs.

### Low-level Python API access

If you want to call low-level functions in the Python C API, you can
do so using `ccall`.  Just remember to call `pyinitialize` first, and:

* Use `pyfunc(func::Symbol)` to get a function pointer to pass to `ccall`
  given a symbol `func` in the Python API.  e.g. you can call `int Py_IsInitialized()` by `ccall(pyfunc(:Py_IsInitialized), Int32, ())`.

* PyCall defines the typealias `PyPtr` for `PythonObject*` argument types,
  and `PythonObject` (see above) arguments are correctly converted to this
  type.

* Use `PythonObject` and the `convert` routines mentioned above to convert
  Julia types to/from `PythonObject*` references.

* If a new reference is returned by a Python function, immediately
  convert the `PyPtr` return values to `PythonObject` objects in order to
  have their Python reference counts decremented when the object is
  garbage collected in Julia.  i.e. `PythonObject(ccall(func, PyPtr, ...))`.

## Author

This package was written by [Steven G. Johnson](http://math.mit.edu/~stevenj/).
