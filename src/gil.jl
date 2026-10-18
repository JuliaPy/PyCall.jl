# GIL management functionality

const _GIL_owner = Threads.Atomic{UInt}(0)

# It is not legal for Julia code to spin directly on "foreign" locks such as
# the GIL, since this can mean never making progress toward.
# Instead we need to tell the julia runtime about it so that GC operations
# can be scheduled.

# Forward declare deferred destructors
function _defer_Py_DecRef end
function _defer_PyBuffer_Release end

# Acquires the GIL.
# This lock can be re-acquired if already held by the OS thread (it is re-entrant).
function GIL_lock()
    @static if VERSION < v"1.12"
        gc_state = ccall(:jl_gc_safe_enter, Int8, ())
        state = ccall((@pysym :PyGILState_Ensure), Cint, ())
        ccall(:jl_gc_safe_leave, Cvoid, (Int8,), gc_state)
        ccall(:jl_gc_safepoint, Cvoid, ())
    elseif libpython == nothing
        state = @ccall gc_safe=true PyGILState_Ensure()::Cint
    else
        state = @ccall gc_safe=true libpython.PyGILState_Ensure()::Cint
    end
    _GIL_owner[] = objectid(current_task())
    return state
end

# Releases the GIL.
# Argument is the state from a corresponding `GIL_lock()` call.
function GIL_unlock(state::Cint)
    _GIL_owner[] = UInt(0)
    ccall((@pysym :PyGILState_Release), Cvoid, (Cint,), state)
end

# Quickly check whether this task holds the GIL.
#
# If true, the current task is guaranteed to be holding the GIL.
# May return false even if the GIL is currently held.
function GIL_held()
    return _GIL_owner[] == objectid(current_task())
end

"""
    @with_GIL expr

Execute `expr` while holding the Python GIL. Safe to nest (re-entrant).

!!! warning "Illegal to yield to the Julia scheduler"

    GIL-locked code MUST NOT access any Julia-side locks or I/O or call
    `yield()`. This is required to avoid task-migration, which would
    leave the GIL held on an "old" OS thread and unable to be released.

    This condition may be relaxed in the future if
    [https://github.com/JuliaLang/julia/issues/52108](https://github.com/JuliaLang/julia/issues/52108)
    is resolved.
"""
macro with_GIL(expr)
    # TODO:
    #   If OS thread pinning is eventually supported in Julia
    #   (c.f. https://github.com/JuliaLang/julia/issues/52108),
    #   replace this with a yield-safe version
    quote
        if GIL_held()
            # Fast path: GIL already held on this OS thread, just run directly
            $(esc(expr))
        else
            local _state = GIL_lock()
            try
                local val = $(esc(expr))
                _check_release_queues()
                val
            finally
                GIL_unlock(_state)
            end
        end
    end
end
