# Interactions of the Julia and Python garbage collection:
#
#    * When we wrap a Python object in a Julia object jo (e.g. a PyDict),
#      we keep an explicit PyObject reference inside jo, whose finalizer
#      decrefs the Python object when it is called.
#
#    * When we wrap a Julia object jo inside a Python object po
#      (e.g a numpy array), we add jo to the pycall_gc dictionary,
#      keyed by a weak reference to po.  The Python weak reference
#      allows us to register a callback function that is called
#      when po is deallocated, and this callback function removes
#      jo from pycall_gc so that Julia can garbage-collect it.

const pycall_gc = Dict{PyPtr,Any}()

function weakref_callback(callback::PyPtr, wo::PyPtr)
    delete!(pycall_gc, wo)
    ccall((@pysym :Py_DecRef), Void, (PyPtr,), wo)
    ccall((@pysym :Py_IncRef), Void, (PyPtr,), pynothing[])
    return pynothing[]
end

const weakref_callback_obj = PyNULL() # weakref_callback Python method

function pygc_finalize()
    pydecref(weakref_callback_obj)
    empty!(pycall_gc)
end

# "embed" a reference to jo in po, using the weak-reference mechanism
function pyembed(po::PyObject, jo::Any)
    if weakref_callback_obj.o == C_NULL
        weakref_callback_obj.o = pyincref(pymethod(weakref_callback,
                                                   "weakref_callback",
                                                   METH_O)).o
    end
    wo = @pycheckn ccall((@pysym :PyWeakref_NewRef), PyPtr, (PyPtr,PyPtr),
                         po, weakref_callback_obj)
    pycall_gc[wo] = jo
    return po
end
