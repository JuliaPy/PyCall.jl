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

using Base.Threads: atomic_add!, atomic_sub!, Atomic, SpinLock

const pycall_gc = Dict{PyPtr,Any}()

function weakref_callback(callback::PyPtr, wo::PyPtr)
    delete!(pycall_gc, wo)
    ccall((@pysym :Py_DecRef), Cvoid, (PyPtr,), wo)
    return pyincref_(pynothing[])
end

const weakref_callback_obj = PyNULL() # weakref_callback Python method

# Python expects the PyMethodDef structure to be a constant, so
# we put it in a global to prevent gc.
const weakref_callback_meth = Ref{PyMethodDef}()

# "embed" a reference to jo in po, using the weak-reference mechanism
function pyembed(po::PyObject, jo::Any)
    # If there's a need to support immutable embedding,
    # the API needs to be changed to return the pointer.
    isimmutable(jo) && throw(ArgumentError("pyembed: immutable argument not allowed"))
    if ispynull(weakref_callback_obj)
        cf = @cfunction(weakref_callback, PyPtr, (PyPtr,PyPtr))
        weakref_callback_meth[] = PyMethodDef("weakref_callback", cf, METH_O)
        copy!(weakref_callback_obj,
              PyObject(@pycheckn ccall((@pysym :PyCFunction_NewEx), PyPtr,
                                       (Ref{PyMethodDef}, Ptr{Cvoid}, Ptr{Cvoid}),
                                       weakref_callback_meth, C_NULL, C_NULL)))
    end
    wo = @pycheckn ccall((@pysym :PyWeakref_NewRef), PyPtr, (PyPtr,PyPtr),
                         po, weakref_callback_obj)
    pycall_gc[wo] = jo
    return po
end

# Deferred `finalizer` / destructor queue(s):
#
#   * In a `finalizer` context, it is unsafe to take the GIL since it
#     can cause deadlocks such as:
#
#       1. Task A holds the GIL and waits for Task B to finish something
#       2. Task B enters GC and tries runs finalizers
#       3. Task B cannot acquire the GIL and Task A is stuck waiting on B
#
#   * To work around this, we defer any GIL-requiring operations to run
#     at the end of next `@with_GIL` block.

const _deferred_count = Atomic{Int}(0)
const _deferred_Py_DecRef = Vector{PyPtr}()
const _deferred_PyBuffer_Release = Vector{PyBuffer}()

# Since it is illegal for finalizers to `yield()` to the Julia scheduler, this
# lock MUST be a `SpinLock` or similar. Most other locks in `Base` implicitly
# yield.
const _deferred_queue_lock = SpinLock()

# Defers a `Py_DecRef(o)` call to be performed later, when the GIL is available
function _defer_Py_DecRef(o::PyObject)
    lock(_deferred_queue_lock)
    try
        push!(_deferred_Py_DecRef, getfield(o, :o))
        atomic_add!(_deferred_count, 1)
    finally
        unlock(_deferred_queue_lock)
    end
    return
end

# Defers a `PyBuffer_Release(o)` call to be performed later, when the GIL is available
function _defer_PyBuffer_Release(o::PyBuffer)
    lock(_deferred_queue_lock)
    try
        push!(_deferred_PyBuffer_Release, o)
        atomic_add!(_deferred_count, 1)
    finally
        unlock(_deferred_queue_lock)
    end
    return
end

# Called at the end of every `@with_GIL` block, this performs any deferred destruction
# operations from finalizers that could not be done immediately due to not holding the GIL
@noinline function _drain_release_queues()
    lock(_deferred_queue_lock)

    atomic_sub!(_deferred_count, length(_deferred_Py_DecRef))
    atomic_sub!(_deferred_count, length(_deferred_PyBuffer_Release))

    Py_DecRefs = copy(_deferred_Py_DecRef)
    PyBuffer_Releases = copy(_deferred_PyBuffer_Release)

    empty!(_deferred_Py_DecRef)
    empty!(_deferred_PyBuffer_Release)

    unlock(_deferred_queue_lock)

    for o in Py_DecRefs
        ccall(@pysym(:Py_DecRef), Cvoid, (PyPtr,), o)
    end
    for o in PyBuffer_Releases
        ccall(@pysym(:PyBuffer_Release), Cvoid, (Ref{PyBuffer},), o)
    end

    return nothing
end
