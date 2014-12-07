# Initializing Python (surprisingly complicated)

#########################################################################

pyconfigvar(python::AbstractString, var::AbstractString) = chomp(readall(`$python -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('$var'))"`))
pysys(python::AbstractString, var::AbstractString) = chomp(readall(`$python -c "import sys; print(sys.$var)"`))
pyconfigvar(python, var, default) = let v = pyconfigvar(python, var)
    v == "None" ? default : v
end

#########################################################################

const dlext = isdefined(Sys, :dlext) ? Sys.dlext : Sys.shlib_ext
const dlprefix = @windows? "" : "lib"

function dlopen_libpython(python::AbstractString)
    # it is ridiculous that it is this hard to find the name of libpython
    v = pyconfigvar(python,"VERSION","")
    libs = [ dlprefix*"python"*v*"."*dlext, dlprefix*"python."*dlext ]
    lib = pyconfigvar(python, "LIBRARY")
    lib != "None" && unshift!(libs, splitext(lib)[1]*"."*dlext)
    lib = pyconfigvar(python, "LDLIBRARY")
    lib != "None" && unshift!(unshift!(libs, basename(lib)), lib)
    libs = unique(libs)

    libpaths = [pyconfigvar(python, "LIBDIR"),
                (@windows ? dirname(pysys(python, "executable")) : joinpath(dirname(dirname(pysys(python, "executable"))), "lib"))]
    @osx_only push!(libpaths, pyconfigvar(python, "PYTHONFRAMEWORKPREFIX"))

    # `prefix` and `exec_prefix` are the path prefixes where python should look for python only and compiled libraries, respectively.
    # These are also changed when run in a virtualenv.
    exec_prefix = pyconfigvar(python, "exec_prefix")
    # Since we only use `libpaths` to find the python dynamic library, we should only add `exec_prefix` to it.
    push!(libpaths, exec_prefix)
    if !haskey(ENV, "PYTHONHOME")
        # PYTHONHOME tells python where to look for both pure python
        # and binary modules.  When it is set, it replaces both
        # `prefix` and `exec_prefix` and we thus need to set it to
        # both in case they differ. This is also what the
        # documentation recommends.  However, they are documented
        # to always be the same on Windows, where it causes
        # problems if we try to include both.
        ENV["PYTHONHOME"] = @windows? exec_prefix : pyconfigvar(python, "prefix") * ":" * exec_prefix
        # Unfortunately, setting PYTHONHOME screws up Canopy's Python distro
        try
	    run(`$python -c "import site"` |> DevNull .> DevNull)
        catch
	    pop!(ENV, "PYTHONHOME")
        end
    end
    # TODO: look in python-config output? pyconfigvar("LDFLAGS")?
    for lib in libs
        for libpath in libpaths
     	    if isfile(joinpath(libpath, lib))
                try
                    return dlopen(joinpath(libpath, lib),
                                  RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
                end
            end
        end
    end

    # We do this *last* because the libpython in the system
    # library path might be the wrong one if multiple python
    # versions are installed (we prefer the one in LIBDIR):
    for lib in libs
        lib = splitext(lib)[1] 
        try
            return dlopen(lib, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL)
        end
    end
    error("Couldn't find libpython; try pyinitialize(\"/path/to/libpython\")")
end

#########################################################################
# Python 3.x uses wchar_t arrays for some string arguments
function wbytestring(s::AbstractString)
    if pyversion.major < 3 || sizeof(Cwchar_t) == 1 # ASCII (or UTF8)
        bytestring(s)
    else # UTF16 or UTF32, presumably
        if sizeof(Cwchar_t) == 4 # UTF32
            n = length(s)
            w = Array(Cwchar_t, n + 1)
            i = 0
            for c in s
                w[i += 1] = c
            end
        else # UTF16, presumably
            @assert sizeof(Cwchar_t) == 2
            if isdefined(Base, :UTF16String) # Julia 0.3
                s16 = utf16(s)
                w = Array(Cwchar_t, length(s.data)+1)
                copy!(w,1, s16.data,1, length(s.data))
            else
                n = length(s)
                w = Array(Cwchar_t, n + 1)
                i = 0
                for c in s
                    # punt on multi-byte encodings
                    c > 0xffff && error("unsupported char $c in \"$s\"")
                    w[i += 1] = c
                end
            end
        end
        w[end] = 0
        w
    end
end

#########################################################################
# global flags to make sure we don't initialize too many times
initialized = false # whether Python is initialized
finalized = false # whether Python has been finalized

# low-level initialization, given a pointer to dlopen result on libpython,
# or C_NULL if python symbols are in the global namespace:
# initialize the Python interpreter (no-op on subsequent calls)
function pyinitialize(libpy::Ptr{Void}, programname="")
    global initialized
    global finalized
    if !initialized::Bool
        if finalized::Bool
            # From the Py_Finalize documentation:
            #    "Some extensions may not work properly if their
            #     initialization routine is called more than once; this
            #     can happen if an application calls Py_Initialize()
            #     and Py_Finalize() more than once."
            # For example, numpy and scipy seem to crash if this is done.
            error("Calling pyinitialize after pyfinalize is not supported")
        end

        # cache the Python version as a Julia VersionNumber
        global const pyversion = getversion(libpy)

        # Py_SetProgramName needs its argument to persist as long as Python
        global const pyprogramname = wbytestring(programname)

        global const libpython = libpy == C_NULL ? ccall(:jl_load_dynamic_library, Ptr{Void}, (Ptr{Uint8},Cuint), C_NULL, 0) : libpy
        already_inited = 0 != ccall((@pysym :Py_IsInitialized), Cint, ())
        if !already_inited
            if !isempty(pyprogramname)
                if pyversion.major < 3
                    ccall((@pysym :Py_SetProgramName), Void, (Ptr{Uint8},), 
                          pyprogramname)
                else
                    ccall((@pysym :Py_SetProgramName), Void, (Ptr{Cwchar_t},), 
                          pyprogramname)
                end
            end
            ccall((@pysym :Py_InitializeEx), Void, (Cint,), 0)
        end
        initialized::Bool = true
        global const inspect = pyimport("inspect")
        global const builtin = pyimport(pyversion.major < 3 ? "__builtin__" : "builtins")

        pyexc_initialize()

        # Python has zillions of types that a function be, in addition
        # to the FunctionType in the C API.  We have to obtain these
        # at runtime and cache them in globals
        types = pyimport("types")
        global const BuiltinFunctionType = types["BuiltinFunctionType"]
        global const TypeType = pybuiltin("type")
        global const MethodType = types["MethodType"]
        global const MethodWrapperType = pytypeof(PyObject(PyObject[])["__add__"])
        global const ufuncType = try
            pyimport("numpy")["ufunc"]
        catch
            PyObject() # NumPy not available
        end

        # PyUnicode_* may actually be a #define for another symbol, so
        # we cache the correct dlsym
        global const PyUnicode_AsUTF8String =
          pysym_e(:PyUnicode_AsUTF8String,
                  :PyUnicodeUCS4_AsUTF8String,
                  :PyUnicodeUCS2_AsUTF8String)
        global const PyUnicode_DecodeUTF8 =
          pysym_e(:PyUnicode_DecodeUTF8,
                  :PyUnicodeUCS4_DecodeUTF8,
                  :PyUnicodeUCS2_DecodeUTF8)

        # cache Python None -- PyPtr, not PyObject, to prevent it from
        # being finalized prematurely on exit
        global const pynothing = convert(PyPtr, pysym(:_Py_NoneStruct))

        # xrange type (or range in Python 3)
        global const pyxrange = pyincref(convert(PyPtr, pysym(:PyRange_Type)))

        # Python 2/3 compatibility: cache dlsym for renamed functions
        if pyhassym(:PyString_FromString)
            global const pystring_fromstring = pysym(:PyString_FromString)
            global const pystring_asstring = pysym(:PyString_AsString)
            global const pystring_size = pysym(:PyString_Size)
            global const pystring_type = pysym(:PyString_Type)
        else
            global const pystring_fromstring = pysym(:PyBytes_FromString)
            global const pystring_asstring = pysym(:PyBytes_AsString)
            global const pystring_size = pysym(:PyBytes_Size)
            global const pystring_type = pysym(:PyBytes_Type)
        end
        if pyhassym(:PyInt_Type)
            global const pyint_type = pysym(:PyInt_Type)
            global const pyint_from_size_t = pysym(:PyInt_FromSize_t)
            global const pyint_from_ssize_t = pysym(:PyInt_FromSsize_t)
            global const pyint_as_ssize_t = pysym(:PyInt_AsSsize_t)
        else
            global const pyint_type = pysym(:PyLong_Type)
            global const pyint_from_size_t = pysym(:PyLong_FromSize_t)
            global const pyint_from_ssize_t = pysym(:PyLong_FromSsize_t)
            global const pyint_as_ssize_t = pysym(:PyLong_AsSsize_t)
        end

        # PyCObject_Check and PyCapsule_CheckExact are actually macros
        # that compare against PyCObject_Type and PyCapsule_Type globals,
        # which we cache if they are available:
        global const PyCObject_Type = pysym_e(:PyCObject_Type)
        global const PyCapsule_Type = pysym_e(:PyCapsule_Type)

        # cache ctypes.c_void_p type and function if available
        vpt, pvp = try
            (pyimport("ctypes")["c_void_p"],
             p::Ptr -> pycall(c_void_p_Type::PyObject, PyObject, uint(p)))
        catch # fallback to CObject
            (pysym(:PyCObject_FromVoidPtr),
             p::Ptr -> PyObject(ccall(pycobject_new, PyPtr, (Ptr{Void}, Ptr{Void}), p, C_NULL)))
        end
        global const c_void_p_Type = vpt
        global const py_void_p = pvp

        # hashes changed from long to intptr_t in Python 3.2
        global const Py_hash_t = pyversion < v"3.2" ? Clong:Int

        # whether to use unicode for strings by default, ala Python 3
        global const pyunicode_literals = pyversion >= v"3.0"

        # traceback.format_tb function, for show(PyError)
        global const format_traceback = pyimport("traceback")["format_tb"]

        # all cfunctions must be compile at runtime
        global const jl_Function_call_ptr =
            cfunction(jl_Function_call, PyPtr, (PyPtr,PyPtr,PyPtr))
        global const pyio_repr_ptr = cfunction(pyio_repr, PyPtr, (PyPtr,))
        global const pyjlwrap_dealloc_ptr = cfunction(pyjlwrap_dealloc, Void, (PyPtr,))
        global const pyjlwrap_repr_ptr = cfunction(pyjlwrap_repr, PyPtr, (PyPtr,))
        global const pyjlwrap_hash_ptr = cfunction(pyjlwrap_hash, Uint, (PyPtr,))
        global const pyjlwrap_hash32_ptr = cfunction(pyjlwrap_hash32, Uint32, (PyPtr,))

        # similarly, any MethodDef calls involve cfunctions
        global const jl_TextIO_methods = make_io_methods(true)
        global const jl_IO_methods = make_io_methods(false)
        global const jl_IO_getset = PyGetSetDef[
            PyGetSetDef("closed", jl_IO_closed)
            PyGetSetDef("encoding", jl_IO_encoding)
            PyGetSetDef()
        ]

        init_datetime()
        pyjlwrap_init()
        
        global const jl_FunctionType = pyjlwrap_type("PyCall.jl_Function",
                                                     t -> t.tp_call =
                                                       jl_Function_call_ptr)

        if !already_inited
            # some modules (e.g. IPython) expect sys.argv to be set
            if pyversion.major < 3
                argv_s = bytestring("")
                argv = convert(Ptr{Uint8}, argv_s)
                ccall(pysym(:PySys_SetArgvEx), Void, (Cint,Ptr{Ptr{Uint8}},Cint), 1, &argv, 0)
            else
                argv_s = Cwchar_t[0]
                argv   = convert(Ptr{Cwchar_t}, argv_s)
                ccall(pysym(:PySys_SetArgvEx), Void, (Cint, Ptr{Ptr{Cwchar_t}}, Cint), 1, &argv, 0)
            end
            
            # Some Python code checks sys.ps1 to see if it is running
            # interactively, and refuses to be interactive otherwise.
            # (e.g. Matplotlib: see PyPlot#79)
            if isinteractive()
                let sys = pyimport("sys")
                    if !haskey(sys, "ps1")
                        sys["ps1"] = ">>> "
                    end
                end
            end
        end
    end
end

# need to be able to get the version before Python is initialized
Py_GetVersion(libpy=libpython) = bytestring(ccall(dlsym(libpy, :Py_GetVersion), Ptr{Uint8}, ()))
getversion(libpy) = convert(VersionNumber, split(Py_GetVersion(libpy))[1])

# initialize the Python interpreter (no-op on subsequent calls)
function pyinitialize(python::AbstractString)
    global initialized
    if !initialized::Bool
        libpy,programname = try
            dlopen_libpython(python), pysys(python, "executable")
        catch
            # perhaps we were passed library name and not executable?
            (dlopen(python, RTLD_LAZY|RTLD_DEEPBIND|RTLD_GLOBAL), python)
        end
        pyinitialize(libpy, programname)
    end
    return
end

pyinitialize() = pyinitialize(get(ENV, "PYTHON", "python")) 
dlopen_libpython() = dlopen_libpython(get(ENV, "PYTHON", "python"))

# end the Python interpreter and free associated memory
function pyfinalize()
    global initialized
    global finalized
    if initialized::Bool
        pygui_stop_all()
        pydecref(mpc)
        pydecref(mpf)
        pydecref(mpmath)
        pydecref(ufuncType)
        npyfinalize()
        pydecref(format_traceback)
        pydecref(c_void_p_Type)
        pydecref(pyxrange)
        pydecref(BuiltinFunctionType)
        pydecref(TypeType)
        pydecref(MethodType)
        pydecref(MethodWrapperType)
        pydecref(builtin)
        pydecref(inspect)
        pyexc_finalize()
        pydecref(jl_FunctionType)
        pygc_finalize()
        gc() # collect/decref any remaining PyObjects
        ccall((@pysym :Py_Finalize), Void, ())
        dlclose(libpython)
        initialized::Bool = false
        finalized::Bool = true
    end
    return
end
