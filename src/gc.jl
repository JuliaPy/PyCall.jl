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

@static if VERSION >= v"1.12"
    using Base.Threads: @atomic, @atomicswap, @atomicreplace
    mutable struct DeferredNode
        obj::Union{PyPtr,PyBuffer}
        @atomic next::Union{DeferredNode,Nothing}
        DeferredNode(x) = new(x, nothing)
    end
    const _deferred_list = DeferredNode(PyPtr(0))

    function _add_deferred(node::DeferredNode)
        head = @atomic :acquire _deferred_list.next
        while true
            @atomic :release node.next = head
            res = @atomicreplace _deferred_list.next head => node
            if res.success
                break
            end
            head = res.old
        end
    end
    function _next_deferred(node::DeferredNode)
        return @atomic :acquire node.next
    end
    function _take_deferred()
        return @atomicswap :acquire _deferred_list.next = nothing
    end
    # For testing only
    function _peek_deferred()
        return @atomic :acquire _deferred_list.next
    end
else
    using Base.Threads: SpinLock
    mutable struct DeferredNode
        obj::Union{PyPtr,PyBuffer}
        next::Union{DeferredNode,Nothing}
        DeferredNode(x) = new(x, nothing)
    end
    # Since it is illegal for finalizers to `yield()` to the Julia scheduler, this
    # lock MUST be a `SpinLock` or similar. Most other locks in `Base` implicitly
    # yield.
    # Also note that the spinlock is not reentrant and since this can be called from
    # the finallizer (i.e. GC) we must not allocate or do anything that can trigger GC
    # in with the lock held. This isn't going to be super reliable
    # but should be good enough to support older julia versions.
    const _deferred_queue_lock = SpinLock()
    const _deferred_list = DeferredNode(PyPtr(0))

    function _add_deferred(node::DeferredNode)
        lock(_deferred_queue_lock) do
            head = _deferred_list.next
            node.next = head
            _deferred_list.next = node
            return
        end
    end
    function _next_deferred(node::DeferredNode)
        return node.next
    end
    function _take_deferred()
        # If the head is nothing, we don't need the load to have any acquire semantics
        # so we could read it out of the locking region as long as the read itself is atomic
        # (i.e. doesn't return torn pointer)
        if _deferred_list.next === nothing
            return
        end
        lock(_deferred_queue_lock) do
            # The head has changed so we need to read again.
            head = _deferred_list.next
            _deferred_list.next = nothing
            return head
        end
    end
    # For testing only
    function _peek_deferred()
        return _deferred_list.next
    end
end

# Defers a `Py_DecRef(o)` call to be performed later, when the GIL is available
function _defer_Py_DecRef(o::PyObject)
    ptr = getfield(o, :o)
    if ptr !== PyPtr_NULL
        _add_deferred(DeferredNode(ptr))
        setfield!(o, :o, PyPtr_NULL)
    end
    return
end

# Defers a `PyBuffer_Release(o)` call to be performed later, when the GIL is available
function _defer_PyBuffer_Release(o::PyBuffer)
    _add_deferred(DeferredNode(o))
    return
end

# Called at the end of every `@with_GIL` block, this performs any deferred destruction
# operations from finalizers that could not be done immediately due to not holding the GIL
@inline function _check_release_queues()
    list = _take_deferred()
    if list === nothing
        return
    end
    _drain_release_queues(list)
end

@noinline function _drain_release_queues(list)
    while list !== nothing
        o = list.obj
        if isa(o, PyBuffer)
            ccall(@pysym(:PyBuffer_Release), Cvoid, (Ref{PyBuffer},), o)
        else
            ccall(@pysym(:Py_DecRef), Cvoid, (PyPtr,), o)
        end
        list = _next_deferred(list)
    end
    return nothing
end
