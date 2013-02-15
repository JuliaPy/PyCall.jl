# Calling Python functions from the Julia language

This package provides a `pycall` function (similar in spirit to
Julia's `ccall` function) to call Python functions from the Julia
language, automatically converting types etcetera.

## Work in Progress

PyCall is currently a proof-of-concept and work in progress.  Some
basic functionality works, but major TODO items are:

* Currently, the Python library name is hardcoded, but this should be
  determined from the `python` executable name given when Python is
  initialized.  For now, change the `libpython` variable in `PyCall.jl`
  to the name of your Python library as needed.

* Support for passing and returning `Array` and `Tuple` arguments, the
  former ideally using [NumPy's](http://www.numpy.org/) interface to
  pass arrays without incurring a copy.

* Currently, Julia needs to be patched to include the `RTLD_GLOBAL` flag
  in `dlopen` (in `dlload.c:jl_uv_dlopen`) in order for it to be possible to
  import modules like [SciPy](http://www.scipy.org) that load their own
  shared libraries.  A more permanent solution needs to be found.

* More error checking and conversion of Python exceptions to Julia exceptions.

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

    require("PyCall")
    using PyCall
    pyinitialize() # initialize the Python interpreter
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

*Currently, only numeric, boolean, and string types are supported, with
more planned.*

You can also look up other names in a module, and use `convert` to
convert them to Julia types, e.g.

    convert(Float64, math["pi"])

returns the numeric value of &pi; from Python's `math.pi`.

## Reference

The following functions are provided by the `PyCall` module:

* `pyinitialize()`: Initialize the Python interpreter.  This `must` be
  called before any of the other functions below, or a crash will occur.
  It is safe to call this function more than once; subsequent calls will
  do nothing.

* `pyfinalize()`: End the Python interpreter and free all associated memory.
  After this function is called, none of the other functions below may
  be called until Python is re-started by calling `pyinitialize()` again.

* `pyimport(s)`: Import the Python module `s` (a string or symbol) and
  return a pointer to it (a `PyObject`).   Functions or other symbols
  in the module may then be looked up by `s[name]` where `name is a string
  or symbol (`s[name]` also returns a `PyObject`).

* `pycall(function::PyObject, returntype::Type, args...)`.   Call the given 
  Python `function` (typically looked up from a module) with the given
  `args...` (of standard Julia types which are converted automatically to
  the corresponding Python types if possible), converting the return value
  to `returntype`.

* `pyglobal(s)`: Look up `s` (a string or symbol) in the global Python
  namespace.

The `PyCall` module also provides a new type `PyObject` (a wrapper around
`PyObject*` in Python's C API) representing a reference to a Python object.

Constructors `PyObject(o)` are provided for a number of Julia types,
and `PyCall` also supplies `convert(T, o::PyObject)` to convert
PyObjects back into Julia types `T`.  *Currently, the only types
supported are numbers (integer, real, and complex), booleans, and
strings, but more are planned.*

## Author

This package was written by [Steven G. Johnson](http://math.mit.edu/~stevenj/).
