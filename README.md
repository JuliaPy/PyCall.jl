# Calling Python functions from the Julia language

[![Build Status](https://travis-ci.org/stevengj/PyCall.jl.svg?branch=master)](https://travis-ci.org/stevengj/PyCall.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/otj9pnwsq32to211?svg=true)](https://ci.appveyor.com/project/StevenGJohnson/pycall-jl)
[![Coverage Status](https://coveralls.io/repos/stevengj/PyCall.jl/badge.svg?branch=master)](https://coveralls.io/r/stevengj/PyCall.jl?branch=master)

[![PyCall](http://pkg.julialang.org/badges/PyCall_0.3.svg)](http://pkg.julialang.org/?pkg=PyCall&ver=0.3)
[![PyCall](http://pkg.julialang.org/badges/PyCall_0.4.svg)](http://pkg.julialang.org/?pkg=PyCall&ver=0.4)
[![PyCall](http://pkg.julialang.org/badges/PyCall_0.5.svg)](http://pkg.julialang.org/?pkg=PyCall&ver=0.5)

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
install the files.  Julia 0.3 or later (0.4 or later is recommended) and Python 2.7 or later are required.

The latest development version of PyCall is avalable from
<https://github.com/stevengj/PyCall.jl>.  If you want to switch to
this after installing the package, run `Pkg.checkout("PyCall"); Pkg.build("PyCall")`.

If a `python` executable is not found (see below), `Pkg.add("PyCall")`
or `Pkg.build("PyCall")` will use the
[Conda.jl](https://github.com/Luthaf/Conda.jl) package to install a
minimal Python distribution (via
[Miniconda](http://conda.pydata.org/docs/install/quick.html)) that is
private to Julia (not in your `PATH`).  You can use the `Conda` Julia
package to install more Python packages, and `import Conda` to print
the `Conda.PYTHONDIR` directory where `python` was installed.

Alternatively, you can make sure Python is already installed before
adding (or building) PyCall, and that a `python` executable is in your
`PATH` (or be specified manually as described below), to make PyCall
link to that version of Python.  Usually, the necessary libraries are
installed along with Python, but [pyenv on
MacOS](https://github.com/stevengj/PyCall.jl/issues/122) requires you
to install it with `env PYTHON_CONFIGURE_OPTS="--enable-framework"
pyenv install 3.4.3`.  The Enthought Canopy Python distribution is
currently [not
supported](https://github.com/stevengj/PyCall.jl/issues/42).
As a general rule, we tend to recommend the [Anaconda Python
distribution](https://store.continuum.io/cshop/anaconda/) on MacOS and
Windows, or using the Julia Conda package, in order to minimize headaches.

## Specifying the Python version

The Python version that is used defaults to whatever `python` program is in
your `PATH`.   If PyCall can't find your Python, then it will install its
own via Conda as described above.

If you want to use a different version of Python on your system, you can change the Python version by setting the `PYTHON` environment variable to the path of the `python` executable and then re-running `Pkg.build("PyCall")`.  In Julia:

    ENV["PYTHON"] = "... path of the python program you want ..."
    Pkg.build("PyCall")

Note also that you will need to re-run `Pkg.build("PyCall")` if your
`python` program changes significantly (e.g. you switch to a new
Python distro, or you switch from Python 2 to Python 3).

To force Julia to use its own Python distribution, via Conda, rather
than whatever is installed on your system, simply set `ENV["PYTHON"]`
to the empty string `""`.

The current Python version being used is stored in the `pyversion`
global variable of the `PyCall` module.  You can also look at
`PyCall.libpython` to find the name of the Python library or
`PyCall.pyprogramname` for the `python` program name.

(Technically, PyCall does not use the `python` program per se: it links
directly to the `libpython` library.  But it finds the location of `libpython`
by running `python` during `Pkg.build`.)

Subsequent builds of PyCall (e.g. when you update the package via
`Pkg.update`) will use the same Python executable name by default,
unless you set the `PYTHON` environment variable or delete the file
`Pkg.dir("PyCall","deps","PYTHON")`.

**Note:** If you use Python
[virtualenvs](http://docs.python-guide.org/en/latest/dev/virtualenvs/),
then be aware that PyCall *uses the virtualenv it was built with*, even
if you switch virtualenvs.  If you want to switch PyCall to use a
different virtualenv, then you should switch virtualenvs and run
`rm(Pkg.dir("PyCall","deps","PYTHON")); Pkg.build("PyCall")`.

## Usage

Here is a simple example to call Python's `math.sin` function and
compare it to the built-in Julia `sin`:

    using PyCall
    @pyimport math
    math.sin(math.pi / 4) - sin(pi / 4)  # returns 0.0

Type conversions are automatically performed for numeric, boolean,
string, IO stream, date/period, and function types, along with tuples,
arrays/lists, and dictionaries of these types. (Python functions can
be converted/passed to Julia functions and vice versa!)  Other types
are supported via the generic PyObject type, below.

Python submodules must be imported by a separate `@pyimport` call, and
in this case you must supply an identifier to to use in Julia.  For example

    @pyimport numpy.random as nr
    nr.rand(3,4)

Multidimensional arrays exploit the NumPy array interface for
conversions between Python and Julia.  By default, they are passed
from Julia to Python without making a copy, but from Python to Julia a
copy is made; no-copy conversion of Python to Julia arrays can be achieved
with the `PyArray` type below.

Keyword arguments can also be passed. For example, matplotlib's
[pyplot](http://matplotlib.org/) uses keyword arguments to specify plot
options, and this functionality is accessed from Julia by:

    @pyimport matplotlib.pyplot as plt
    x = linspace(0,2*pi,1000); y = sin(3*x + 4*cos(2*x));
    plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
    plt.show()

However, for better integration with Julia graphics backends and to
avoid the need for the `show` function, we recommend using matplotlib
via the Julia [PyPlot module](https://github.com/stevengj/PyPlot.jl).

Arbitrary Julia functions can be passed to Python routines taking
function arguments.  For example, to find the root of cos(x) - x,
we could call the Newton solver in scipy.optimize via:

    @pyimport scipy.optimize as so
    so.newton(x -> cos(x) - x, 1)

**Important:** The biggest difference from Python is that object attributes/members are
accessed with `o[:attribute]` rather than `o.attribute`, so that `o.method(...)` in
Python is replaced by `o[:method](...)` in Julia.  Also, you use
`get(o, key)` rather than `o[key]`.  (However, you can access integer
indices via `o[i]` as in Python, albeit with 1-based Julian indices rather
than 0-based Python indices.)  (This is because Julia does not
permit overloading the `.` operator yet.)  See also the section on
`PyObject` below, as well as the `pywrap` function to create anonymous
modules that simulate `.` access (this is what `@pyimport` does).  For
example, using [Biopython](http://biopython.org/wiki/Seq) we can do:

    @pyimport Bio.Seq as s
    @pyimport Bio.Alphabet as a
    my_dna = s.Seq("AGTACACTGGT", a.generic_dna)
    my_dna[:find]("ACT")

whereas in Python the last step would have been `my_dna.find("ACT")`.

## Troubleshooting

Here are solutions to some common problems:

* As mentioned above, use `foo[:bar]` and `foo[:bar](...)` rather than `foo.bar` and `foo.bar(...)`,
respectively, to access attributes and methods of Python objects.

* In Julia 0.3, sometimes calling a Python function fails because PyCall doesn't realize it is a callable object (since so many types of objects can be callable in Python).   The workaround is to use `pycall(foo, PyAny, args...)` instead of `foo(args...)`.   If you want to call `foo.bar(args...)` in Python, it is good to use `pycall(foo["bar"], PyAny, args...)`, where using `foo["bar"]` instead of `foo[:bar]` prevents any automatic conversion of the `bar` field.  In Julia 0.4, however, this problem goes away: all PyObjects are automatically callable, thanks to call overloading in Julia 0.4.

* By default, PyCall [doesn't include the current directory in the Python search path](https://github.com/stevengj/PyCall.jl/issues/48).   If you want to do that (in order to load a Python module from the current directory), just run `unshift!(PyVector(pyimport("sys")["path"]), "")`.

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
PyObjects back into Julia types `T`.  Currently, the types supported
are numbers (integer, real, and complex), booleans, strings, IO streams,
dates/periods, and functions, along with tuples and arrays/lists
thereof, but more are planned.  (Julia symbols are converted to Python
strings.)

Given `o::PyObject`, `o[:attribute]` is equivalent to `o.attribute`
in Python, with automatic type conversion.  To get an attribute as a
`PyObject` without type conversion, do `o["attribute"]` instead.

Given `o::PyObject`, `get(o, key)` is equivalent to `o[key]` in
Python, with automatic type conversion.  To get as a `PyObject`
without type conversion, do `get(o, PyObject, key)`, or more generally
`get(o, SomeType, key)`.  You can also supply a default value to use
if the key is not found by `get(o, key, default)` or `get(o, SomeType,
key, default)`.  Similarly, `set!(o, key, val)` is equivalent to
`o[key] = val` in Python, and `delete!(o, key)` is equivalent to `del
o[key]` in Python.   For one or more *integer* indices, `o[i]` in Julia
is equivalent to `o[i-1]` in Python.

In Julia 0.4, you can call an `o::PyObject` via `o(args...)` just like
in Python (assuming that the object is callable in Python).  In Julia
0.3, you have to do `pycall(o, PyAny, args...)`, although the explicit
`pycall` form is still useful in Julia 0.4 if you want to specify the
return type.

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
e.g. when passed as `pycall` arguments.

#### PyVector

The PyCall module provides a new type `PyVector` (a subclass of
`AbstractVector`) which implements a no-copy wrapper around an
arbitrary Python list or sequence object.  (Unlike `PyArray`, the
`PyVector` type is not limited to `NumPy` arrays, although using
`PyArray` for the latter is generally more efficient.)  Just use
`PyArray` as the return type of a `pycall` returning a list or
sequence object (including tuples), or call `PyVector(o::PyObject)` on
a sequence object `o`.

A `v::PyVector` supports the usual `v[index]` referencing and assignment,
along with `delete!` and `pop!` operations.  `copy(v)` converts `v` to
an ordinary Julia `Vector`.

#### PyDict

Similar to `PyVector`, PyCall also provides a type `PyDict` (a subclass
of `Association`) that implements a no-copy wrapper around a Python
dictionary (or any object implementing the mapping protocol).  Just
use `PyDict` as the return type of a `pycall` returning a dictionary,
or call `PyDict(o::PyObject)` on a dictionary object `o`.  By
default, a `PyDict` is an `Any => Any` dictionary (or actually `PyAny
=> PyAny`) that performs runtime type inference, but if your Python
dictionary has known, fixed types you can insteady use `PyDict{K,V}` given
the key and value types `K` and `V` respectively.

Currently, passing Julia dictionaries to Python makes a copy of the Julia
dictionary.

#### PyTextIO

Julia `IO` streams are converted into Python objects implementing the
[RawIOBase](http://docs.python.org/2/library/io.html#io.RawIOBase)
interface, so they can be used for binary I/O in Python.  However,
some Python code (notably unpickling) expects a stream implementing
the
[TextIOBase](http://docs.python.org/2/library/io.html#io.TextIOBase)
interface, which differs from RawIOBase mainly in that `read` an
`readall` functions return strings rather than byte arrays.  If you
need to pass an `IO` stream as a text-IO object, just call
`PyTextIO(io::IO)` to convert it.

(There doesn't seem to be any good way to determine automatically
whether Python wants a stream argument to return strings or binary
data.  Also, unlike Python, Julia does not open files separately in
"text" or "binary" modes, so we cannot determine the conversion simply
from how the file was opened.)

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
  return a pointer to it (a `PyObject`).  Functions or other symbols
  in the module may then be looked up by `s[name]` where `name` is a
  string (for the raw `PyObject`) or symbol (for automatic
  type-conversion).  Unlike the `@pyimport` macro, this does not
  define a Julia module and members cannot be accessed with `s.name`.

* `pyeval(s::AbstractString, rtype=PyAny; locals...)` evaluates `s`
  as a Python string and returns the result converted to `rtype`
  (which defaults to `PyAny`).  The remaining arguments are keywords
  that define local variables to be used in the expression.  For
  example, `pyeval("x + y", x=1, y=2)` returns `3`.

* `pybuiltin(s)`: Look up `s` (a string or symbol) among the global Python
  builtins.  If `s` is a string it returns a `PyObject`, while if `s` is a
  symbol it returns the builtin converted to `PyAny`.

* `pywrap(o::PyObject)` returns a wrapper `w` that is an anonymous
  module which provides (read) access to converted versions of `o`'s
  members as `w.member`.  (For example, `@pyimport module as name` is
  equivalent to `name = pywrap(pyimport("module"))`.)  If the Python
  module contains identifiers that are reserved words in Julia
  (e.g. `function`), they cannot be accessed as `w.member`; one must
  instead use `w.pymember(:member)` (for the `PyAny` conversion) or
  `w.pymember("member")` (for the raw `PyObject`).

Occasionally, you may need to pass a keyword argument to Python that
is a [reserved word](https://en.wikipedia.org/wiki/Reserved_word) in Julia.
For example, calling `f(x, function=g)` will fail because `function` is
a reserved word in Julia. In such cases, you can use the lower-level
Julia syntax `f(x; :function=>g)`.

### GUI Event Loops

For Python packages that have a graphical user interface (GUI),
notably plotting packages like matplotlib (or MayaVi or Chaco), it is
convenient to start the GUI event loop (which processes things like
mouse clicks) as an asynchronous task within Julia, so that the GUI is
responsive without blocking Julia's input prompt.  PyCall includes
functions to implement these event loops for some of the most common
cross-platform [GUI
toolkits](http://en.wikipedia.org/wiki/Widget_toolkit):
[wxWidgets](http://www.wxwidgets.org/), [GTK+](http://www.gtk.org/) version 2 (via [PyGTK](http://www.pygtk.org/)) or version 3 (via [PyGObject](https://wiki.gnome.org/action/show/Projects/PyGObject)),
and [Qt](http://qt-project.org/) (via the [PyQt4](http://wiki.python.org/moin/PyQt4) or [PySide](http://qt-project.org/wiki/PySide)
Python modules).

You can set a GUI event loop via:

* `pygui_start(gui::Symbol=pygui())`.  Here, `gui` is either `:wx`,
  `:gtk`, `:gtk3`, `:tk`, or `:qt` to start the respective toolkit's
  event loop.  (`:qt` will use PyQt4 or PySide, preferring the former;
  if you need to require one or the other you can instead use
  `:qt_pyqt4` or `:qt_pyside`, respectively.) It defaults to the
  return value of `pygui()`, which returns a current default GUI (see
  below).  Passing a `gui` argument also changes the default GUI,
  equivalent to calling `pygui(gui)` below.  You may start event loops
  for more than one GUI toolkit (to run simultaneously).  Calling
  `pygui_start` more than once for a given toolkit does nothing
  (except to change the current `pygui` default).

* `pygui()`: return the current default GUI toolkit (`Symbol`).  If
  the default GUI has not been set already, this is the first of
  `:tk`, `:qt`, `:wx`, `:gtk`, or `:gtk3` for which the corresponding
  Python package is installed.  `pygui(gui::Symbol)` changes the
  default GUI to `gui`.

* `pygui_stop(gui::Symbol=pygui())`: Stop any running event loop for `gui`
  (which defaults to the current return value of `pygui`).  Returns
  `true` if an event loop was running, and `false` otherwise.

To use these GUI facilities with some Python libraries, it is enough
to simply start the appropriate toolkit's event-loop before importing
the library.  However, in other cases it is necessary to explicitly
tell the library which GUI toolkit to use and that an interactive mode
is desired.  To make this even easier, it is convenient to have
wrapper modules around popular Python libraries, such as the [PyPlot
module](https://github.com/stevengj/PyPlot.jl) for Julia.

### Low-level Python API access

If you want to call low-level functions in the Python C API, you can
do so using `ccall`.

* Use `@pysym(func::Symbol)` to get a function pointer to pass to `ccall`
  given a symbol `func` in the Python API.  e.g. you can call `int Py_IsInitialized()` by `ccall(@pysym(:Py_IsInitialized), Int32, ())`.

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

* The function `pyerr_check(msg::AbstractString)` can be used to check if a
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
  given a Python type object `t`.  `pytypeof(o::PyObject)` returns the
  Python type of `o`, equivalent to `type(o)` in Python.

### Using PyCall from Julia Modules

You can use PyCall from any Julia code, including within Julia modules. However, some care is required when using PyCall from [precompiled Julia modules](http://docs.julialang.org/en/latest/manual/modules/#module-initialization-and-precompilation). The key thing to remember is that *all Python objects* (any `PyObject`) contain *pointers* to memory allocated by the Python runtime, and such pointers *cannot be saved* in precompiled constants.   (When a precompiled library is reloaded, these pointers will not contain valid memory addresses.)

The solution is fairly simple:

* Python objects that you create in functions called *after* the module is loaded are always safe.

* If you want to store a Python object in a global variable that is initialized automatically when the module is loaded, then initialize the variable in your module's `__init__` function.  For a type-stable global constant, initialize the constant to `PyCall.PyNULL()` at the top level, and then use the `copy!` function in your module's `__init__` function to mutate it to its actual value.

For example, suppose your module uses the `scipy.optimize` module, and you want to load this module when your module is loaded and store it in a global constant `scipy_opt`.  You could do:

```jl
__precompile__() # this module is safe to precompile
module MyModule
using PyCall

const scipy_opt = PyCall.PyNULL()

function __init__()
    copy!(scipy_opt, pyimport("scipy.optimize"))
end

end
```
Then you can access the `scipy.optimize` functions as `scipy_opt[:newton]`
and so on.

(Note that you cannot use `@pyimport` safely with precompilation, because
that declares a global constant that internally has a pointer to the module.)

## Author

This package was written by [Steven G. Johnson](http://math.mit.edu/~stevenj/).
