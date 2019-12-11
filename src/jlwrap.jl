################################################################
# Wrap a Python type around a Julia Any object

const GLOBAL_CACHE = Base.IdSet()

function cached(x)
    push!(GLOBAL_CACHE, x)
    x
end

cached_string_pointer(x::String) = unsafe_convert(Ptr{UInt8}, cached(x))
cached_string_pointer(x::AbstractString) = cached_string_pointer(String(x))
cached_string_pointer(x::Ptr) = convert(Ptr{UInt8}, x)

cached_string_pointer_or_NULL(x::AbstractString) =
    isempty(x) ? Ptr{UInt8}(C_NULL) : cached_string_pointer(x)
cached_string_pointer_or_NULL(x::Ptr) = cached_string_pointer(x)

cached_ref(x) = cached(Ref(x))
cached_ref(x::Ref) = cached(x)

cached_ref_pointer(x) = cached_ref_pointer(Ref(x))
cached_ref_pointer(x::Ref) = unsafe_convert(Ptr{eltype(x)}, cached(x))

CPyMethodDef(; name=C_NULL, meth=C_NULL, flags=0, doc=C_NULL) =
    CPyMethodDef(cached_string_pointer(name), convert(Ptr{Cvoid}, meth), convert(Cint, flags), cached_string_pointer_or_NULL(doc))

CPyGetSetDef(; name=C_NULL, get=C_NULL, set=C_NULL, doc=C_NULL, closure=C_NULL) =
    CPyGetSetDef(cached_string_pointer(name), convert(Ptr{Cvoid}, get), convert(Ptr{Cvoid}, set), cached_string_pointer_or_NULL(doc), convert(Ptr{Cvoid}, closure))

CPyMemberDef(; name=C_NULL, typ=0, offset=0, flags=0, doc=C_NULL) =
    CPyMemberDef(cached_string_pointer(name), convert(Cint, typ), convert(Int, offset), convert(Cint, flags), cached_string_pointer_or_NULL(doc))

@eval CPyNumberMethods(; $([Expr(:kw, n, C_NULL) for n in fieldnames(CPyNumberMethods)]...)) = CPyNumberMethods($([:(convert(Ptr{Cvoid}, $n)) for n in fieldnames(CPyNumberMethods)]...))

@eval CPySequenceMethods(; $([Expr(:kw, n, C_NULL) for n in fieldnames(CPySequenceMethods)]...)) = CPySequenceMethods($([:(convert(Ptr{Cvoid}, $n)) for n in fieldnames(CPySequenceMethods)]...))

@eval CPyMappingMethods(; $([Expr(:kw, n, C_NULL) for n in fieldnames(CPyMappingMethods)]...)) = CPyMappingMethods($([:(convert(Ptr{Cvoid}, $n)) for n in fieldnames(CPyMappingMethods)]...))

@eval function CPyTypeObject(; initialize=true, $([Expr(:kw, n, t<:Ptr ? C_NULL : 0) for (n,t) in zip(fieldnames(CPyTypeObject), fieldtypes(CPyTypeObject))]...))
    # convert inputs
    if tp_name isa AbstractString
        tp_name = cached_string_pointer(tp_name)
    end
    if tp_as_number isa CPyNumberMethods
        tp_as_number = cached_ref_pointer(tp_as_number)
    end
    if tp_as_sequence isa CPySequenceMethods
        tp_as_sequence = cached_ref_pointer(tp_as_sequence)
    end
    if tp_as_mapping isa CPyMappingMethods
        tp_as_mapping = cached_ref_pointer(tp_as_mapping)
    end
    if tp_members isa AbstractVector{CPyMemberDef}
        tp_members = pointer(cached([tp_members; CPyMemberDef_NULL]))
    end
    if tp_methods isa AbstractVector{CPyMethodDef}
        tp_methods = pointer(cached([tp_methods; CPyMethodDef_NULL]))
    end
    if tp_getset isa AbstractVector{CPyGetSetDef}
        tp_getset = pointer(cached([tp_getset; CPyGetSetDef_NULL]))
    end
    if tp_base isa Ref{CPyTypeObject}
        tp_base = cached_ref_pointer(tp_base)
    end
    # make the type
    CPyTypeObject($([:(convert($t, $n)) for (n,t) in zip(fieldnames(CPyTypeObject), fieldtypes(CPyTypeObject))]...))
end

macro cpymethod(name)
    :(@cfunction($name, PyPtr, (PyPtr, PyPtr)))
end

macro cpygetfunc(name)
    :(@cfunction($name, PyPtr, (PyPtr, Ptr{Cvoid})))
end

# TODO: fully implement all methods for collections.abc
# TODO: fully implement all methods for io

struct CPyJlWrapObject
    # PyObject_HEAD (for non-Py_TRACE_REFS build):
    ob_refcnt::Int
    ob_type::PyPtr # actually Ptr{PyTypeObject}

    ob_weakrefs::PyPtr
    jl_value::Any
end

const sizeof_CPyJlWrapObject_HEAD = sizeof_CPyObject_HEAD + sizeof(PyPtr)

# destructor for jlwrap instance, assuming it was created with pyjlwrap_new
function _pyjlwrap_dealloc(o::PyPtr)
    p = convert(Ptr{PyPtr}, o)
    if unsafe_load(p, 3) != PyPtr_NULL
        PyObject_ClearWeakRefs(o)
    end
    delete!(pycall_gc, o)
    return nothing
end

unsafe_pyjlwrap_to_objref(o::Union{PyPtr, PyObject}) =
  GC.@preserve o unsafe_pointer_to_objref(unsafe_load(convert(Ptr{Ptr{Cvoid}}, PyPtr(o)), 4))

pyerrorval(::Type{PyPtr}) = PyPtr_NULL
pyerrorval(::Type{T}) where {T<:Integer} = zero(T) - one(T)

macro pyjlwrapfunc(ex)
    def = MacroTools.splitdef(ex)
    def[:name] = Symbol(:_pyjlwrap_, def[:name])
    selfarg = def[:args][1]
    (selfname, selftype, selfslurp, selfdefault) = MacroTools.splitarg(selfarg)
    _self = gensym()
    err = gensym()
    def[:args][1] = :($_self :: PyPtr)
    def[:body] = quote
        $selfname :: $selftype = unsafe_pyjlwrap_to_objref($_self)
        try
            $(def[:body])
        catch $err
            @pyraise $err
        end
        $(pyerrorval(eval(def[:rtype])))
    end
    r = MacroTools.combinedef(def)
end

function _pyjlwrap_repr(_o::PyPtr)::PyPtr
    try
        if _o == C_NULL
            @pyreturn "<PyCall.JlWrap NULL>"
        else
            o = unsafe_pyjlwrap_to_objref(_o)
            n = unsafe_string(unsafe_load(unsafe_pytype(_o)).tp_name)
            @pyreturn "<$n $(repr(o))>"
        end
    catch e
        @pyraise e
    end
    return PyPtr_NULL
end

function _pyjlwrap_str(_o::PyPtr)::PyPtr
    try
        if _o == C_NULL
            @pyreturn "NULL"
        else
            o = unsafe_pyjlwrap_to_objref(_o)
            @pyreturn string(o)
        end
    catch e
        @pyraise e
    end
    return PyPtr_NULL
end

function _pyjlwrap_hash(o::PyPtr)::UInt
    h = hash(unsafe_pyjlwrap_to_objref(o))
    # Python hashes are not permitted to return -1!!
    return h == reinterpret(UInt, -1) ? pysalt::UInt : h::UInt
end

# 32-bit hash on 64-bit machines, needed for Python < 3.2 with Windows
const pysalt32 = 0xb592cd9b # hash("PyCall") % UInt32
function _pyjlwrap_hash32(o::PyPtr)::UInt32
    h = ccall(:int64to32hash, UInt32, (UInt64,),
              hash(unsafe_pyjlwrap_to_objref(o)))
    # Python hashes are not permitted to return -1!!
    return h == reinterpret(UInt32, Int32(-1)) ? pysalt32 : h::UInt32
end

function _pyjlwrap_call(f_::PyPtr, args_::PyPtr, kw_::PyPtr)::PyPtr
    f = unsafe_pyjlwrap_to_objref(f_)
    args = PyObject(args_) # don't need pyincref because of finally clause below
    try
        jlargs = julia_args(f, args)

        # we need to use invokelatest to get execution in newest world
        if kw_ == C_NULL
            ret = Base.invokelatest(f, jlargs...)
        else
            kw = PyDict{Symbol,PyObject}(pyincref(kw_))
            kwargs = [ (k,julia_kwarg(f,k,v)) for (k,v) in kw ]

            # 0.6 `invokelatest` doesn't support kwargs, instead
            # use a closure over kwargs. see:
            #   https://github.com/JuliaLang/julia/pull/22646
            f_kw_closure() = f(jlargs...; kwargs...)
            ret = Core._apply_latest(f_kw_closure)
        end

        return pyreturn(ret)
    catch e
        @pyraise e
    finally
        setfield!(args, :o, PyPtr_NULL) # don't decref
    end
    return PyPtr_NULL
end

@pyjlwrapfunc function length(o)::Cssize_t
    return length(o)
end

@pyjlwrapfunc function istrue(o)::Cint
    return _pyistrue(o)::Bool
end

_pyistrue(x) =
    try
        !iszero(x)
    catch
        try
            !isempty(x)
        catch
            true
        end
    end
_pyistrue(::Nothing) = false
_pyistrue(::Missing) = false
_pyistrue(x::Bool) = x
_pyistrue(x::Number) = !iszero(x)
_pyistrue(x::Union{AbstractArray,AbstractDict,Tuple,Pair,NamedTuple,AbstractSet}) = !isempty(x)
_pyistrue(x::Symbol) = x != Symbol()
_pyistrue(x::Ptr) = x != C_NULL
_pyistrue(x::Ref) = true

@pyjlwrapfunc function int(o)::PyPtr
    @pyreturn convert(Integer, o)
end

@pyjlwrapfunc function float(o)::PyPtr
    @pyreturn convert(AbstractFloat, o)
end

function _pyjlwrap_richcompare(a::PyPtr, b::PyPtr, op::Cint)::PyPtr
    a = unsafe_pyjlwrap_to_objref(a)
    b = convert(PyAny, pyincref(b))
    b isa PyObject && @pyreturn_NotImplemented
    x = try
        op == Py_LT ? a < b :
        op == Py_LE ? a ≤ b :
        op == Py_EQ ? a == b :
        op == Py_NE ? a != b :
        op == Py_GT ? a > b :
        op == Py_GE ? a ≥ b :
        error("invalid op")
    catch e
        if e isa MethodError
            @pyreturn_NotImplemented
        else
            @pyraise e
            @pyreturn_NULL
        end
    end
    @pyreturn x
end

@pyjlwrapfunc function getitem(o, i::PyPtr)::PyPtr
    @pyreturn _pygetitem(o, pyincref(i))
end

@pyjlwrapfunc function getitem_oneup(o, i::Cssize_t)::PyPtr
    @pyreturn o[i+1]
end

@pyjlwrapfunc function getitem_oneup(o, _i::PyPtr)::PyPtr
    i = convert(Union{Int,Tuple{Vararg{Int}}}, pyincref(_i)) .+ 1
    @pyreturn o[i...]
end

@pyjlwrapfunc function setitem_oneup(o, _i::PyPtr, _x::PyPtr)::Cint
    i = convert(Union{Int, Tuple{Vararg{Int}}}, pyincref(_i)) .+ 1
    o[i...] = _pyvalueatindex(o, i, _x)
    return 0
end

@pyjlwrapfunc function getitem_namedtuple(o, _i::PyPtr)::PyPtr
    i = convert(Union{Int,Symbol}, pyincref(_i))
    if i isa Int
        @pyreturn o[i+1]
    else
        @pyreturn o[i]
    end
end

_pygetitem(o, i::PyObject) = _pygetitem(o, _pyindex(o, i))
_pygetitem(o, i) = getindex(o, i)
_pygetitem(o, i::Tuple) = applicable(getindex, o, i...) ? getindex(o, i...) : getindex(o, i)

function _pyindex(o, i)
    T = applicable(keytype, o) ? keytype(o) : Any
    T = T==Any ? PyAny : Union{T,PyAny}
    convert(T, i)
end
_pyindex(o::NamedTuple, i::PyObject) = convert(Union{Symbol,Int}, i)

@pyjlwrapfunc function setitem(o, i::PyPtr, x::PyPtr)::Cint
    _pysetitem(o, pyincref(i), pyincref(x))
    return 0
end

function _pysetitem(o, i, x)
    i = _pyindex(o, i)
    x = _pyvalueatindex(o, i, x)
    setindex!(o, x, i)
end

function _pyvalueatindex(o, i, x)
    T = applicable(eltype, o) ? eltype(o) : Any
    T = T==Any ? PyAny : Union{T,PyAny}
    convert(T, x)
end

docstring(x) = string(Docs.doc(x))

function _pyjlwrap_getattr(self_::PyPtr, attr__::PyPtr)::PyPtr
    attr_ = PyObject(attr__) # don't need pyincref because of finally clause below
    try
        self = unsafe_pyjlwrap_to_objref(self_)
        attr = convert(String, attr_)
        if startswith(attr, "__julia_field_")
            a = Symbol(attr[15:end])
            if hasfield(typeof(self), a)
                @pyreturn getfield(self, a)
            end
        elseif startswith(attr, "__julia_property_")
            a = Symbol(attr[18:end])
            if hasproperty(typeof(self), a)
                @pyreturn getproperty(self, a)
            end
        else
            a = Symbol(attr)
            if hasproperty(self, a)
                @pyreturn getproperty(self, a)
            end
        end
        return ccall(@pysym(:PyObject_GenericGetAttr), PyPtr, (PyPtr, PyPtr), self_, attr__)
        # if startswith(attr, "__")
        #     if attr in ("__name__","func_name")
        #         return pystealref!(PyObject(string(f)))
        #     elseif attr in ("__doc__", "func_doc")
        #         return pystealref!(PyObject(docstring(f)))
        #     elseif attr in ("__module__","__defaults__","func_defaults","__closure__","func_closure")
        #         return pystealref!(PyObject(nothing))
        #     elseif startswith(attr, "__jlfield_")
        #         return pyreturn(getfield(f, Symbol(attr[11:end])))
        #     else
        #         # TODO: handle __code__/func_code (issue #268)
        #         return PyObject_GenericGetAttr(self_, attr__)
        #     end
        # else
            # fidx = Base.fieldindex(typeof(f), Symbol(attr), false)
            # if fidx != 0
            #     return pyreturn(getfield(f, fidx))
            # else
            #     return ccall(@pysym(:PyObject_GenericGetAttr), PyPtr, (PyPtr,PyPtr), self_, attr__)
            # end
            # return pyreturn(getproperty(f, Symbol(attr)))
        # end
    catch e
        @pyraise e
    finally
        setfield!(attr_, :o, PyPtr_NULL) # don't decref
    end
    return PyPtr_NULL
end

function _pyjlwrap_setattr(self_::PyPtr, attr__::PyPtr, value_::PyPtr)::Cint
    value_ == C_NULL && return pyjlwrap_delattr(self_, attr__)
    attr_ = PyObject(attr__)
    value = pyincref(value_)
    try
        self = unsafe_pyjlwrap_to_objref(self_)
        attr = convert(String, attr_)
        @show self attr
        if startswith(attr, "__")
            if startswith(attr, "__julia_field_")
                _pysetfield(self, Symbol(attr[15:end]), value)
            elseif startswith(attr, "__julia_property_")
                _pysetproperty(self, Symbol(attr[18:end]), value)
            else
                return PyObject_GenericSetAttr(self_, attr__, value_)
            end
        else
            _pysetproperty(self, Symbol(attr), value)
        end
        return 0
    catch e
        @show e
        @pyraise e
    finally
        setfield!(attr_, :o, PyPtr_NULL)
    end
    return -1
end

function _pysetproperty(o, f, x)
    x = _pyvalueatproperty(o, f, x)
    setproperty!(o, f, x)
end

function _pyvalueatproperty(o, f, x)
    convert(PyAny, x)
end

function _pysetfield(o, f, x)
    x = _pyvalueatfield(o, f, x)
    setfield!(o, f, x)
end

function _pyvalueatfield(o, f, x)
    T = fieldtype(typeof(o), f)
    T = T==Any ? PyAny : T
    convert(T, x)
end

# tp_iternext object of a jlwrap_iterator object, similar to PyIter_Next
@pyjlwrapfunc function iternext(self)::PyPtr
    iter, iter_result_ref = self
    iter_result = iter_result_ref[]
    if iter_result !== nothing
        item, state = iter_result
        iter_result_ref[] = iterate(iter, state)
        return pyreturn(item)
    end
end

# the tp_iter slot of jlwrap object: like PyObject_GetIter, it
# returns a reference to a new jlwrap_iterator object
@pyjlwrapfunc function getiter(self)::PyPtr
    return pystealref!(pyjlwrap_iterator(self))
end

@pyjlwrapfunc function getiter_keys(self)::PyPtr
    return pystealref!(pyjlwrap_iterator(keys(self)))
end

for (a,b) in [(:negative,:-), (:positive, :+), (:absolute, :abs), (:invert, :~)]
    @eval @pyjlwrapfunc function $a(self)::PyPtr
        return pystealref!(pyjlwrap($b(self)))
    end
end

for (a,b) in [(:add, :+), (:subtract, :-), (:multiply, :*), (:remainder, :mod), (:lshift, :(<<)), (:rshift, :(>>)), (:and, :&), (:xor, :⊻), (:or, :|), (:floordivide, :fld), (:truedivide, :/)]
    @eval function $(Symbol(:_pyjlwrap_, a))(a_::PyPtr, b_::PyPtr)::PyPtr
        (is_pyjlwrap(a_) && is_pyjlwrap(b_)) || @pyreturn_NotImplemented
        try
            a = unsafe_pyjlwrap_to_objref(a_)
            b = unsafe_pyjlwrap_to_objref(b_)
            return pystealref!(pyjlwrap($b(a, b)))
        catch e
            @pyraise e
        end
        PyPtr_NULL
    end
end

function _pyjlwrap_power(_a::PyPtr, _b::PyPtr, _c::PyPtr)
    if _c == pynothing[]
        (is_pyjlwrap(_a) && is_pyjlwrap(_b)) || @pyreturn_NotImplemented
        try
            a = unsafe_pyjlwrap_to_objref(_a)
            b = unsafe_pyjlwrap_to_objref(_b)
            return pystealref!(pyjlwrap(a^b))
        catch e
            @pyraise e
        end
        PyPtr_NULL
    else
        @pyreturn_NotImplemented
    end
end

@pyjlwrapfunc function get_real(self, ::Ptr{Cvoid})::PyPtr
    @pyreturn real(self)
end

@pyjlwrapfunc function get_imag(self, ::Ptr{Cvoid})::PyPtr
    @pyreturn imag(self)
end

@pyjlwrapfunc function conjugate(self, ::PyPtr)::PyPtr
    @pyreturn conj(self)
end

@pyjlwrapfunc function trunc(self, ::PyPtr)::PyPtr
    @pyreturn trunc(Integer, self)
end

@pyjlwrapfunc function round(self, ::PyPtr)::PyPtr
    @pyreturn round(Integer, self)
end

@pyjlwrapfunc function floor(self, ::PyPtr)::PyPtr
    @pyreturn floor(Integer, self)
end

@pyjlwrapfunc function ceil(self, ::PyPtr)::PyPtr
    @pyreturn ceil(Integer, self)
end

@pyjlwrapfunc function get_numerator(self, ::Ptr{Cvoid})::PyPtr
    @pyreturn numerator(self)
end

@pyjlwrapfunc function get_denominator(self, ::Ptr{Cvoid})::PyPtr
    @pyreturn denominator(self)
end

@pyjlwrapfunc function io_close(self, ::PyPtr)::PyPtr
    close(self)
    @pyreturn nothing
end

@pyjlwrapfunc function io_get_closed(self, ::Ptr{Cvoid})::PyPtr
    @pyreturn !isopen(self)
end

@pyjlwrapfunc function io_flush(self, ::PyPtr)::PyPtr
    flush(self)
    @pyreturn nothing
end

@pyjlwrapfunc function io_isatty(self, ::PyPtr)::PyPtr
    @pyreturn (self isa Base.TTY)
end

@pyjlwrapfunc function io_readable(self, ::PyPtr)::PyPtr
    @pyreturn isreadable(self)
end

@pyjlwrapfunc function io_writable(self, ::PyPtr)::PyPtr
    @pyreturn iswritable(self)
end

@pyjlwrapfunc function io_tell(self, ::PyPtr)::PyPtr
    @pyreturn position(self)
end

@pyjlwrapfunc function io_write_str(self, x::PyPtr)::PyPtr
    @pyreturn write(self, pystr(pyincref(x)))
end

# Given a jlwrap type, create a new instance (and save value for gc)
function pyjlwrap_new(pyT::Ref{CPyTypeObject}, value::Any)
    o = PyObject(@pycheckn CPyObject_New(pyT))
    # o = PyObject(@pycheckn ccall((@pysym :_PyObject_New),
    #                              PyPtr, (Ref{CPyTypeObject},), pyT))
    p = convert(Ptr{Ptr{Cvoid}}, PyPtr(o))
    if isimmutable(value)
        # It is undefined to call `pointer_from_objref` on immutable objects.
        # The compiler is free to return basically anything since the boxing is not
        # significant at all.
        # Below is a well defined way to get a pointer (`ptr`) and an object that defines
        # the lifetime of the pointer `ref`.
        ref = Ref{Any}(value)
        pycall_gc[PyPtr(o)] = ref
        ptr = unsafe_load(Ptr{Ptr{Cvoid}}(pointer_from_objref(ref)))
    else
        pycall_gc[PyPtr(o)] = value
        ptr = pointer_from_objref(value)
    end
    unsafe_store!(p, C_NULL, 3)
    unsafe_store!(p, ptr, 4)
    return o
end

is_pyjlwrap(o::Union{PyObject,PyPtr}) = CPyJlWrap_Type[].tp_new != C_NULL && CPyObject_IsInstance(o, CPyJlWrap_Type) == 1 #ccall((@pysym :PyObject_IsInstance), Cint, (PyPtr, Ref{CPyTypeObject}), o, CPyJlWrap_Type) == 1

const pyjlwrap_membername = "__jlvalueptr"
const pyjlwrap_doc = "Julia jl_value_t* (Any object)"

# base type
const CPyJlWrap_Type = Ref(CPyTypeObject_NULL)
# for iterators (a tuple `(x, iterate(x))`)
# abstract base classes from `numbers`
const CPyJlWrapNumber_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapComplex_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapReal_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapRational_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapIntegral_Type = Ref(CPyTypeObject_NULL)
# abstract base classes from `collections.abc`
const CPyJlWrapIterable_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapIterator_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapContainer_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapCollection_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapSequence_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapMutableSequence_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapByteString_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapSet_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapMutableSet_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapMapping_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapJlNamedTuple_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapMutableMapping_Type = Ref(CPyTypeObject_NULL)
# abstract base classes from `io`
const CPyJlWrapIOBase_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapRawIO_Type = Ref(CPyTypeObject_NULL)
const CPyJlWrapTextIO_Type = Ref(CPyTypeObject_NULL)

const CPyNumberMethods_default = Ref(CPyNumberMethods_NULL)
const CPyMappingMethods_default = Ref(CPyMappingMethods_NULL)
const CPySequenceMethods_oneup = Ref(CPySequenceMethods_NULL)
const CPyMappingMethods_oneup = Ref(CPyMappingMethods_NULL)
const CPyMappingMethods_namedtuple = Ref(CPyMappingMethods_NULL)

const have_stackless_extension = Ref(false)

function pyjlwraptype_defaultflags()
    flags =
        if pyversion.major ≥ 3
             Py_TPFLAGS_HAVE_VERSION_TAG
        else
            Py_TPFLAGS_HAVE_GETCHARBUFFER |
            Py_TPFLAGS_HAVE_SEQUENCE_IN |
            Py_TPFLAGS_HAVE_INPLACEOPS |
            Py_TPFLAGS_HAVE_RICHCOMPARE |
            Py_TPFLAGS_HAVE_WEAKREFS |
            Py_TPFLAGS_HAVE_ITER |
            Py_TPFLAGS_HAVE_CLASS |
            Py_TPFLAGS_HAVE_INDEX
        end
    if have_stackless_extension[]
        flags |= Py_TPFLAGS_HAVE_STACKLESS_EXTENSION
    end
    flags
end

function pyjlwrap_init()

    empty!(GLOBAL_CACHE)

    # detect at runtime whether we are using Stackless Python
    try
        pyimport("stackless")
        have_stackless_extension[] = true
    catch
        have_stackless_extension[] = false
    end

    CPyNumberMethods_default[] = CPyNumberMethods(
        nb_bool = @cfunction(_pyjlwrap_istrue, Cint, (PyPtr,)),
        nb_int = @cfunction(_pyjlwrap_int, PyPtr, (PyPtr,)),
        nb_float = @cfunction(_pyjlwrap_float, PyPtr, (PyPtr,)),
        nb_negative = @cfunction(_pyjlwrap_negative, PyPtr, (PyPtr,)),
        nb_positive = @cfunction(_pyjlwrap_positive, PyPtr, (PyPtr,)),
        nb_absolute = @cfunction(_pyjlwrap_absolute, PyPtr, (PyPtr,)),
        nb_invert = @cfunction(_pyjlwrap_invert, PyPtr, (PyPtr,)),
        nb_add = @cfunction(_pyjlwrap_add, PyPtr, (PyPtr, PyPtr)),
        nb_subtract = @cfunction(_pyjlwrap_subtract, PyPtr, (PyPtr, PyPtr)),
        nb_multiply = @cfunction(_pyjlwrap_multiply, PyPtr, (PyPtr, PyPtr)),
        nb_remainder = @cfunction(_pyjlwrap_remainder, PyPtr, (PyPtr, PyPtr)),
        nb_lshift = @cfunction(_pyjlwrap_lshift, PyPtr, (PyPtr, PyPtr)),
        nb_rshift = @cfunction(_pyjlwrap_rshift, PyPtr, (PyPtr, PyPtr)),
        nb_and = @cfunction(_pyjlwrap_and, PyPtr, (PyPtr, PyPtr)),
        nb_xor = @cfunction(_pyjlwrap_xor, PyPtr, (PyPtr, PyPtr)),
        nb_or = @cfunction(_pyjlwrap_or, PyPtr, (PyPtr, PyPtr)),
        nb_floordivide = @cfunction(_pyjlwrap_floordivide, PyPtr, (PyPtr, PyPtr)),
        nb_truedivide = @cfunction(_pyjlwrap_truedivide, PyPtr, (PyPtr, PyPtr)),
        nb_power = @cfunction(_pyjlwrap_power, PyPtr, (PyPtr, PyPtr, PyPtr)),
    )

    CPyMappingMethods_default[] = CPyMappingMethods(
        mp_length = @cfunction(_pyjlwrap_length, Cssize_t, (PyPtr,)),
        mp_subscript = @cfunction(_pyjlwrap_getitem, PyPtr, (PyPtr, PyPtr)),
        mp_ass_subscript = @cfunction(_pyjlwrap_setitem, Cint, (PyPtr, PyPtr, PyPtr)),
    )

    CPySequenceMethods_oneup[] = CPySequenceMethods(
        sq_length = @cfunction(_pyjlwrap_length, Cssize_t, (PyPtr,)),
        sq_item = @cfunction(_pyjlwrap_getitem_oneup, PyPtr, (PyPtr, Cssize_t)),
    )

    CPyMappingMethods_oneup[] = CPyMappingMethods(
        mp_length = @cfunction(_pyjlwrap_length, Cssize_t, (PyPtr,)),
        mp_subscript = @cfunction(_pyjlwrap_getitem_oneup, PyPtr, (PyPtr, PyPtr)),
        mp_ass_subscript = @cfunction(_pyjlwrap_setitem_oneup, Cint, (PyPtr, PyPtr, PyPtr)),
    )

    CPyMappingMethods_namedtuple[] = CPyMappingMethods(
        mp_subscript = @cfunction(_pyjlwrap_getitem_namedtuple, PyPtr, (PyPtr, PyPtr)),
    )

    CPyJlWrap_Type[] = CPyTypeObject(
        tp_name = "PyCall.JlWrap",
        tp_basicsize = sizeof(CPyJlWrapObject),
        tp_new = @pyglobal(:PyType_GenericNew),
        tp_flags = pyjlwraptype_defaultflags() | Py_TPFLAGS_BASETYPE,
        tp_members = [
            CPyMemberDef(name="__julia_value", typ=T_PYSSIZET, offset=sizeof_CPyJlWrapObject_HEAD, flags=READONLY),
        ],
        tp_dealloc = @cfunction(_pyjlwrap_dealloc, Cvoid, (PyPtr,)),
        tp_repr = @cfunction(_pyjlwrap_repr, PyPtr, (PyPtr,)),
        tp_str = @cfunction(_pyjlwrap_str, PyPtr, (PyPtr,)),
        tp_call = @cfunction(_pyjlwrap_call, PyPtr, (PyPtr,PyPtr,PyPtr)),
        tp_getattro = @cfunction(_pyjlwrap_getattr, PyPtr, (PyPtr,PyPtr)),
        tp_setattro = @cfunction(_pyjlwrap_setattr, Cint, (PyPtr,PyPtr,PyPtr)),
        tp_iter = @cfunction(_pyjlwrap_getiter, PyPtr, (PyPtr,)),
        tp_hash = sizeof(Py_hash_t) < sizeof(Int) ?
            @cfunction(_pyjlwrap_hash32, UInt32, (PyPtr,)) :
            @cfunction(_pyjlwrap_hash, UInt, (PyPtr,)),
        tp_weaklistoffset = fieldoffset(CPyJlWrapObject, 3),
        tp_richcompare = @cfunction(_pyjlwrap_richcompare, PyPtr, (PyPtr,PyPtr,Cint)),
        tp_as_number = CPyNumberMethods_default[],
        tp_as_mapping = CPyMappingMethods_default[],
    )
    @pycheckz CPyType_Ready(CPyJlWrap_Type)
    CPy_IncRef(CPyJlWrap_Type)

    # # ABSTRACT BASE CLASSES FROM `numbers`

    # CPyJlWrapNumber_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapNumber",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrap_Type,
    # )

    # CPyJlWrapComplex_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapComplex",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapNumber_Type,
    #     tp_getset = [
    #         CPyGetSetDef(name="real", get=@cpygetfunc(_pyjlwrap_get_real)),
    #         CPyGetSetDef(name="imag", get=@cpygetfunc(_pyjlwrap_get_imag)),
    #     ],
    #     tp_methods = [
    #         CPyMethodDef(name="conjugate", meth=@cpymethod(_pyjlwrap_conjugate), flags=METH_NOARGS),
    #     ]
    # )

    # CPyJlWrapReal_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapReal",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapComplex_Type,
    #     tp_methods = [
    #         CPyMethodDef(name="trunc", meth=@cpymethod(_pyjlwrap_trunc), flags=METH_NOARGS),
    #         CPyMethodDef(name="round", meth=@cpymethod(_pyjlwrap_round), flags=METH_NOARGS),
    #         CPyMethodDef(name="floor", meth=@cpymethod(_pyjlwrap_floor), flags=METH_NOARGS),
    #         CPyMethodDef(name="ceil", meth=@cpymethod(_pyjlwrap_ceil), flags=METH_NOARGS),
    #     ]
    # )

    # CPyJlWrapRational_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapRational",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapReal_Type,
    #     tp_getset = [
    #         CPyGetSetDef(name="numerator", get=@cpygetfunc(_pyjlwrap_get_numerator)),
    #         CPyGetSetDef(name="denominator", get=@cpygetfunc(_pyjlwrap_get_denominator)),
    #     ],
    # )

    # CPyJlWrapIntegral_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapIntegral",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapRational_Type,
    # )

    # # ABSTRACT BASE CLASSES FROM `collections.abc`

    # CPyJlWrapIterable_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapIterable",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrap_Type,
    # )

    # CPyJlWrapIterator_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapIterator",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrap_Type,
    #     tp_iter = @cfunction(pyincref_, PyPtr, (PyPtr,)),
    #     tp_iternext = @cfunction(_pyjlwrap_iternext, PyPtr, (PyPtr,)),
    # )
    # Py_IncRef(CPyJlWrap_Type)

    # CPyJlWrapContainer_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapContainer",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrap_Type,
    # )

    # CPyJlWrapCollection_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapCollection",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapIterable_Type,
    # )

    # CPyJlWrapSequence_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapSequence",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapCollection_Type,
    #     tp_as_sequence = CPySequenceMethods_oneup[],
    #     tp_as_mapping = CPyMappingMethods_oneup[],
    # )

    # CPyJlWrapMutableSequence_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapMutableSequence",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapSequence_Type,
    # )

    # CPyJlWrapByteString_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapByteString",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapSequence_Type,
    # )

    # CPyJlWrapSet_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapSet",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapCollection_Type,
    # )

    # CPyJlWrapMutableSet_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapMutableSet",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapSet_Type,
    # )

    # CPyJlWrapMapping_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapMapping",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapCollection_Type,
    #     tp_as_mapping = CPyMappingMethods_default[],
    #     tp_iter = @cfunction(_pyjlwrap_getiter_keys, PyPtr, (PyPtr,)),        
    # )

    # CPyJlWrapJlNamedTuple_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapJlNamedTuple",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapMapping_Type,
    #     tp_as_sequence = CPySequenceMethods_oneup[],
    #     tp_as_mapping = CPyMappingMethods_namedtuple[],
    # )

    # CPyJlWrapMutableMapping_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapMutableMapping",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapMapping_Type,
    # )

    # # ABSTRACT BASE CLASSES FROM `io`

    # CPyJlWrapIOBase_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapIOBase",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrap_Type,
    #     tp_methods = [
    #         CPyMethodDef(name="close", meth=@cpymethod(_pyjlwrap_io_close), flags=METH_NOARGS),
    #         CPyMethodDef(name="flush", meth=@cpymethod(_pyjlwrap_io_flush), flags=METH_NOARGS),
    #         CPyMethodDef(name="isatty", meth=@cpymethod(_pyjlwrap_io_isatty), flags=METH_NOARGS),
    #         CPyMethodDef(name="readable", meth=@cpymethod(_pyjlwrap_io_readable), flags=METH_NOARGS),
    #         CPyMethodDef(name="writable", meth=@cpymethod(_pyjlwrap_io_writable), flags=METH_NOARGS),
    #         CPyMethodDef(name="tell", meth=@cpymethod(_pyjlwrap_io_tell), flags=METH_NOARGS),
    #     ]
    # )

    # CPyJlWrapRawIO_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapRawIO",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapIOBase_Type,
    # )

    # CPyJlWrapTextIO_Type[] = CPyTypeObject(
    #     tp_name = "PyCall.JlWrapTextIO",
    #     tp_basicsize = sizeof(CPyJlWrapObject),
    #     tp_new = @pyglobal(:PyType_GenericNew),
    #     tp_flags = pyjlwraptype_defaultflags(),
    #     tp_base = CPyJlWrapIOBase_Type,
    #     tp_getset = [
    #         CPyGetSetDef(name="closed", get=@cpygetfunc(_pyjlwrap_io_get_closed)),
    #     ],
    #     tp_methods = [
    #         CPyMethodDef(name="write", get=@cpymethod(_pyjlwrap_io_write_str), METH_O),
    #     ],
    # )

    # m = pyimport("numbers")
    # m.Number.register(PyObject(CPyJlWrapNumber_Type))
    # m.Complex.register(PyObject(CPyJlWrapComplex_Type))
    # m.Real.register(PyObject(CPyJlWrapReal_Type))
    # m.Rational.register(PyObject(CPyJlWrapRational_Type))
    # m.Integral.register(PyObject(CPyJlWrapIntegral_Type))

    # m = pyimport("collections.abc")
    # m.Iterable.register(PyObject(CPyJlWrapIterable_Type))
    # m.Iterator.register(PyObject(CPyJlWrapIterator_Type))
    # m.Container.register(PyObject(CPyJlWrapContainer_Type))
    # m.Collection.register(PyObject(CPyJlWrapCollection_Type))
    # m.Sequence.register(PyObject(CPyJlWrapSequence_Type))
    # m.MutableSequence.register(PyObject(CPyJlWrapMutableSequence_Type))
    # m.ByteString.register(PyObject(CPyJlWrapByteString_Type))
    # m.Set.register(PyObject(CPyJlWrapSet_Type))
    # m.MutableSet.register(PyObject(CPyJlWrapMutableSet_Type))
    # m.Mapping.register(PyObject(CPyJlWrapMapping_Type))
    # m.MutableMapping.register(PyObject(CPyJlWrapMutableMapping_Type))

    # m = pyimport("io")
    # m.IOBase.register(PyObject(CPyJlWrapIOBase_Type))
    # m.RawIOBase.register(PyObject(CPyJlWrapRawIO_Type))
    # m.TextIOBase.register(PyObject(CPyJlWrapTextIO_Type))

end

PyObject(t::Ref{CPyTypeObject}) = pyincref(Base.unsafe_convert(PyPtr, t))
PyObject(x::Any) = pyjlwrap(x)

export pyjlwrap, pyjlwrap_textio, pyjlwrap_rawio
pyjlwrap(x) = pyjlwrap_new(CPyJlWrap_Type, x)
# pyjlwrap(x::Union{AbstractDict,AbstractArray,AbstractSet,NamedTuple,Tuple}) = pyjlwrap_iterable(x)
# pyjlwrap(x::Number) = pyjlwrap_number(x)
# pyjlwrap(x::IO) = pyjlwrap_io(x)

# pyjlwrap_iterator(o) =
#     let it = iterate(o)
#         pyjlwrap_new(CPyJlWrapIterator_Type, (o, Ref{Any}(it)))
#     end

# pyjlwrap_number(x) = pyjlwrap_new(CPyJlWrapNumber_Type, x)
# pyjlwrap_number(x::Complex) = pyjlwrap_complex(x)
# pyjlwrap_number(x::Real) = pyjlwrap_real(x)

# pyjlwrap_complex(x) = pyjlwrap_new(CPyJlWrapComplex_Type, x)
# pyjlwrap_complex(x::Real) = pyjlwrap_real(x)

# pyjlwrap_real(x) = pyjlwrap_new(CPyJlWrapReal_Type, x)
# pyjlwrap_real(x::Integer) = pyjlwrap_integral(x)
# pyjlwrap_real(x::Rational) = pyjlwrap_rational(x)

# pyjlwrap_rational(x) = pyjlwrap_new(CPyJlWrapRational_Type, x)
# pyjlwrap_rational(x::Integer) = pyjlwrap_integral(x)

# pyjlwrap_integral(x) = pyjlwrap_new(CPyJlWrapIntegral_Type, x)

# pyjlwrap_iterable(o) = pyjlwrap_new(CPyJlWrapIterable_Type, o)
# pyjlwrap_iterable(o::Union{AbstractDict,AbstractArray,AbstractSet,NamedTuple,Tuple}) = pyjlwrap_collection(o)

# pyjlwrap_collection(o) = pyjlwrap_new(CPyJlWrapCollection_Type, o)
# pyjlwrap_collection(o::Union{Tuple,AbstractArray}) = pyjlwrap_sequence(o)
# pyjlwrap_collection(o::AbstractSet) = pyjlwrap_set(o)
# pyjlwrap_collection(o::Union{AbstractDict,NamedTuple}) = pyjlwrap_mapping(o)

# pyjlwrap_sequence(o) = pyjlwrap_new(CPyJlWrapSequence_Type, o)
# pyjlwrap_sequence(o::AbstractArray) = pyjlwrap_mutablesequence(o)

# pyjlwrap_mutablesequence(o) = pyjlwrap_new(CPyJlWrapMutableSequence_Type, o)

# pyjlwrap_mapping(o) = pyjlwrap_new(CPyJlWrapMapping_Type, o)
# pyjlwrap_mapping(o::Base.ImmutableDict) = pyjlwrap_new(CPyJlWrapMapping_Type, o)
# pyjlwrap_mapping(o::AbstractDict) = pyjlwrap_mutablemapping(o)
# pyjlwrap_mapping(o::NamedTuple) = pyjlwrap_namedtuple(o)

# pyjlwrap_namedtuple(o) = pyjlwrap_new(CPyJlWrapJlNamedTuple_Type, o)

# pyjlwrap_mutablemapping(o) = pyjlwrap_new(CPyJlWrapMutableMapping_Type, o)

# pyjlwrap_set(o) = pyjlwrap_new(CPyJlWrapSet_Type, o)
# pyjlwrap_set(o::AbstractSet) = pyjlwrap_mutableset(o)

# pyjlwrap_mutableset(o) = pyjlwrap_new(CPyJlWrapMutableSet_Type, o)

# pyjlwrap_io(o) = pyjlwrap_new(CPyJlWrapIOBase_Type, o)
# pyjlwrap_io(o::IO) = pyjlwrap_rawio(o)

# pyjlwrap_rawio(o) = pyjlwrap_new(CPyJlWrapRawIO_Type, o)

# pyjlwrap_textio(o) = pyjlwrap_new(CPyJlWrapTextIO_Type, o)
