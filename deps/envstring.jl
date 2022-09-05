mutable struct EnvString <: AbstractString
    _envname::String
    _val::String
    _default::Union{String, Function}
    _ptr::Ptr{Cvoid}

    function EnvString(envname::String, default::AbstractString)
        this = new()
        this._envname = envname
        this._default = default
        this._ptr = C_NULL
        return this
    end

    function EnvString(f::Function, envname::String)
        this = new()
        this._envname = envname
        this._default = f
        this._ptr = C_NULL
        return this
    end
end

Base.show(io::IO, envstr::EnvString) = Base.show(io, loadenvstring(envstr))
Base.print(io::IO, envstr::EnvString) = Base.print(io, loadenvstring(envstr))
Base.println(io::IO, envstr::EnvString) = Base.println(io, loadenvstring(envstr))
Base.ncodeunits(envstr::EnvString) = ncodeunits(loadenvstring(envstr))
Base.isvalid(envstr::EnvString) = isvalid(loadenvstring(envstr))
Base.isvalid(envstr::EnvString, i::Integer) = isvalid(loadenvstring(envstr), i)
Base.iterate(envstr::EnvString) = iterate(loadenvstring(envstr))
Base.iterate(envstr::EnvString, state::Integer) = iterate(loadenvstring(envstr), state)
Base.String(envstr::EnvString) = loadenvstring(envstr)
Base.string(envstr::EnvString) = loadenvstring(envstr)

function loadenvstring(envstr::EnvString)
    envstr._ptr != C_NULL && isdefined(envstr, :_val) && return envstr._val
    src = envstr._val =
        if envstr._default isa Function
            get(ENV, envstr._envname) do
                envstr._default()
            end
        else
            get(ENV, envstr._envname, envstr._default)
        end
    envstr._ptr = Ptr{Cvoid}(1234)
    return src
end

function setenvstring(envstr::EnvString, val::AbstractString)
    envstr._val = ENV[envstr._envname] = String(val)
    envstr._ptr = Ptr{Cvoid}(1234)
end
