using Conda

function PyPreferences.status()
    # TODO: compare with in-process values
    code = """
    $(load_pypreferences_code())
    PyPreferences.Implementations.status_inprocess()
    """
    cmd = include_stdin_cmd()
    open(pipeline(cmd; stdout = stdout, stderr = stderr); write = true) do io
        write(io, code)
    end
    return
end

function PyPreferences.use_system(python::AbstractString = "python3")
    """
    Use python from a provided executable or path (defaults to `python3`).
    """
    return Implementations.set(python = python)
end

function PyPreferences.use_conda()
    """
    Use Python provided by Conda.jl
    """
    Conda.add("numpy")
    return Implementations.set(conda = true)
end

#=
function use_jll()
end
=#

function PyPreferences.use_inprocess()
    return Implementations.set(inprocess = true)
end

function PyPreferences.instruction_message()
    return """
    PyPreferences.jl is not configured properly. Run:
        using Pkg
        Pkg.add("PyPreferences")
        using PyPreferences
        @doc PyPreferences
    for usage.
    """
end


function PyPreferences.assert_configured()
    if (
        PyPreferences.python === nothing ||
        PyPreferences.python_fullpath === nothing ||
        PyPreferences.libpython === nothing ||
        PyPreferences.python_version === nothing ||
        PyPreferences.PYTHONHOME === nothing
    )
        error(PyPreferences.instruction_message())
    end
end