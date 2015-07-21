# Conversion functions for date and time objects in the Python datetime
# module and the Julia Dates module.

# Dates is built-in in Julia 0.4
if !isdefined(Base, :Dates)
    import Dates
end

# Unfortunately, the Python C API (in Python/Include/datetime.h) is somewhat
# painful to call from Julia because it consists mainly of macros that
# translate to lookups in datastructures that are loaded at runtime by
# a PyDateTime_IMPORT macro.  So, we have to mirror those declarations here.
# Fortunately, they don't seem to have changed much since Python 2.7, with
# the biggest difference being the use of a 64-bit hash type.

immutable PyDateTime_CAPI
    # type objects:
    DateType::PyPtr
    DateTimeType::PyPtr
    TimeType::PyPtr
    DeltaType::PyPtr
    TZInfoType::PyPtr

    # function pointers:
    Date_FromDate::Ptr{Void}
    DateTime_FromDateAndTime::Ptr{Void}
    Time_FromTime::Ptr{Void}
    Delta_FromDelta::Ptr{Void}
    DateTime_FromTimestamp::Ptr{Void}
    Date_FromTimestamp::Ptr{Void}
end

immutable PyDateTime_Delta{H} # H = Clong in Python 2, Py_hash_t in Python 3
    # PyObject_HEAD (for non-Py_TRACE_REFS build):
    ob_refcnt::Int
    ob_type::PyPtr
    hashcode::H
    days::Cint
    seconds::Cint
    microseconds::Cint
end

# called from __init__
function init_datetime()
    # emulate PyDateTime_IMPORT:
    global const PyDateTimeAPI =
        unsafe_load(@pycheckn ccall((@pysym :PyCapsule_Import),
                                     Ptr{PyDateTime_CAPI}, (Ptr{Uint8}, Cint),
                                     "datetime.datetime_CAPI", 0))

    # the DateTime, Date, and Time objects are a struct with fields:
    #     ob_refcnt::Int
    #     ob_type::PyPtr
    #     hashcode::Py_hash_t
    #     hastzinfo::Uint8
    #     unsigned char data[LEN]
    #     tzinfo::PyPtr
    # where LEN = 4 for Date, 6 for Time, and 10 for DateTime.  We
    # will access this via raw Ptr{Uint8} loads, with the following offset
    # for data:
    global const PyDate_HEAD = sizeof(Int)+sizeof(PyPtr)+sizeof(Py_hash_t)+1
end

PyObject(d::Dates.Date) =
    PyObject(@pycheckn ccall(PyDateTimeAPI.Date_FromDate, PyPtr,
                             (Cint, Cint, Cint, PyPtr),
                             Dates.year(d), Dates.month(d), Dates.day(d),
                             PyDateTimeAPI.DateType))

PyObject(d::Dates.DateTime) =
    PyObject(@pycheckn ccall(PyDateTimeAPI.DateTime_FromDateAndTime, PyPtr,
                             (Cint, Cint, Cint, Cint, Cint, Cint, Cint,
                              PyPtr, PyPtr),
                             Dates.year(d), Dates.month(d), Dates.day(d),
                             Dates.hour(d), Dates.minute(d), Dates.second(d),
                             Dates.millisecond(d) * 1000,
                             pynothing, PyDateTimeAPI.DateTimeType))

PyDelta_FromDSU(days, seconds, useconds) =
    PyObject(@pycheckn ccall(PyDateTimeAPI.Delta_FromDelta, PyPtr,
                             (Cint, Cint, Cint, Cint, PyPtr),
                             days, seconds, useconds,
                             1, PyDateTimeAPI.DeltaType))

PyObject(p::Dates.Day) = PyDelta_FromDSU(@compat(Int(p)), 0, 0)

function PyObject(p::Dates.Second)
    # normalize to make Cint overflow less likely
    s = @compat Int(p)
    d = div(s, 86400)
    s -= d * 86400
    PyDelta_FromDSU(d, s, 0)
end

function PyObject(p::Dates.Millisecond)
    # normalize to make Cint overflow less likely
    ms = @compat Int(p)
    s = div(ms, 1000)
    ms -= s * 1000
    d = div(s, 86400)
    s -= d * 86400
    PyDelta_FromDSU(d, s, ms * 1000)
end

for T in (:Date, :DateTime, :Delta)
    f = symbol(string("Py", T, "_Check"))
    t = Expr(:., :PyDateTimeAPI, QuoteNode(symbol(string(T, "Type"))))
    @eval $f(o::PyObject) = pyisinstance(o, $t)
end

function pydate_query(o::PyObject)
    if PyDate_Check(o)
        return PyDateTime_Check(o) ? Dates.DateTime : Dates.Date
    elseif PyDelta_Check(o)
        return Dates.Millisecond
    else
        return None
    end
end

function convert(::Type{Dates.DateTime}, o::PyObject)
    if PyDate_Check(o)
        dt = convert(Ptr{Uint8}, o.o) + PyDate_HEAD
        if PyDateTime_Check(o)
            Dates.DateTime(@compat(UInt(unsafe_load(dt,1))<<8)|unsafe_load(dt,2), # Y
                           unsafe_load(dt,3), unsafe_load(dt,4), # month, day
                           unsafe_load(dt,5), unsafe_load(dt,6), # hour, minute
                           unsafe_load(dt,7), # second
                           div((@compat(UInt(unsafe_load(dt,8)) << 16)) |
                               (@compat(UInt(unsafe_load(dt,9)) << 8)) |
                               unsafe_load(dt,10), 1000)) # μs ÷ 1000
        else
            Dates.DateTime(@compat(UInt(unsafe_load(dt,1))<<8)|unsafe_load(dt,2), # Y
                           unsafe_load(dt,3), unsafe_load(dt,4)) # month, day
        end
    else
        throw(ArgumentError("unknown DateTime type $o"))
    end
end

function convert(::Type{Dates.Date}, o::PyObject)
    if PyDate_Check(o)
        dt = convert(Ptr{Uint8}, o.o) + PyDate_HEAD
        Dates.Date(@compat(UInt(unsafe_load(dt,1)) << 8) | unsafe_load(dt,2), # Y
                   unsafe_load(dt,3), unsafe_load(dt,4)) # month, day
    else
        throw(ArgumentError("unknown Date type $o"))
    end
end

function delta_dsμ(o::PyObject)
    PyDelta_Check(o) || throw(ArgumentError("$o is not a timedelta instance"))
    p = unsafe_load(convert(Ptr{PyDateTime_Delta{Py_hash_t}}, o.o))
    return (p.days, p.seconds, p.microseconds)
end

# Should we throw an InexactError when converting a period
# that is not an exact multiple of the resulting unit?   For now,
# follow the lead of Dates and truncate; see Julia issue #9169.

function convert(::Type{Dates.Millisecond}, o::PyObject)
    (d,s,μs) = delta_dsμ(o)
    return Dates.Millisecond((86400d + s) * 1000 + div(μs, 1000))
end

function convert(::Type{Dates.Second}, o::PyObject)
    (d,s,μs) = delta_dsμ(o)
    return Dates.Second(86400d + s + div(μs, 1000000))
end

function convert(::Type{Dates.Day}, o::PyObject)
    (d,s,μs) = delta_dsμ(o)
    return Dates.Day(d + div(s + div(μs, 1000000), 86400))
end
