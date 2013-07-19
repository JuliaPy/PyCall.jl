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

pycall_gc = Dict{PyPtr,Any}()

function weakref_callback(callback::PyPtr, wo::PyPtr)
    global pycall_gc
    try
        delete!(pycall_gc::Dict{PyPtr,Any}, wo)
        # not sure what to do if there is an exception here
    finally
        ccall((@pysym :Py_DecRef), Void, (PyPtr,), wo)
    end
    ccall((@pysym :Py_IncRef), Void, (PyPtr,), pynothing::PyPtr)
    return pynothing::PyPtr
end

weakref_callback_obj = PyObject() # weakref_callback Python method

function pygc_finalize()
    global pycall_gc
    global weakref_callback_obj
    pydecref(weakref_callback_obj)
    pycall_gc::Dict{PyPtr,Any} = Dict{PyPtr,Any}()
end

# "embed" a reference to jo in po, using the weak-reference mechanism
function pyembed(po::PyObject, jo::Any)
    global pycall_gc
    global weakref_callback_obj
    if (weakref_callback_obj::PyObject).o == C_NULL
        weakref_callback_obj::PyObject = pymethod(weakref_callback,
                                                  "weakref_callback",
                                                  METH_O)
    end
    wo = @pycheckn ccall((@pysym :PyWeakref_NewRef), PyPtr, (PyPtr,PyPtr), 
                         po, weakref_callback_obj)
    (pycall_gc::Dict{PyPtr,Any})[wo] = jo
    return po
end